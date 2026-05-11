import cv2
import torch
import numpy as np
import heapq
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def astar(grid, start, goal):
    """A* Pathfinding Algorithm"""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
            
        for dy, dx in neighbors:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_score = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return None

def main(video_path):
    # Load Models
    model = YOLO('yolov8n.pt')
    print("Initializing DeepSORT...")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    
    print("Loading MiDaS model...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    GRID_ROWS = 30
    GRID_COLS = 40

    # Track depth history to calculate relative velocity
    track_history = {} # track_id -> list of depths

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read the frame.")
            break

        H, W = frame.shape[:2]
        annotated_frame = frame.copy()

        # 1. Object Detection
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        # 2. Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        # 3. Depth Estimation
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)

        # 4. Obstacle Grid, Risk, and Relative Velocity
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        resized_depth = cv2.resize(depth_map_normalized, (GRID_COLS, GRID_ROWS))
        grid[resized_depth > 180] = 1

        max_risk_score = 0
        most_critical_depth_change = 0
        current_tracked_ids = []

        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            current_tracked_ids.append(track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            region = depth_map_normalized[max(0, center_y-5):min(H, center_y+5), max(0, center_x-5):min(W, center_x+5)]
            avg_depth = np.median(region) if region.size > 0 else 0
            
            # Record history for velocity
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(avg_depth)
            if len(track_history[track_id]) > 5:
                track_history[track_id].pop(0)

            # Calculate depth change: positive means getting closer, negative means pulling away
            depth_change = 0
            if len(track_history[track_id]) >= 2:
                depth_change = avg_depth - np.mean(track_history[track_id][:-1])

            # Map object to grid
            gx1, gy1 = int((x1 / W) * GRID_COLS), int((y1 / H) * GRID_ROWS)
            gx2, gy2 = int((x2 / W) * GRID_COLS), int((y2 / H) * GRID_ROWS)
            grid[gy1:gy2+1, gx1:gx2+1] = 1
            
            # Annotate tracking
            color = [int(c) for c in colors[int(track_id) % len(colors)]]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Show depth change (rel velocity)
            vel_str = f"{depth_change:+.1f}"
            cv2.putText(annotated_frame, f"ID:{track_id} Prox:{avg_depth:.0f} RelV:{vel_str}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate collision risk (objects in center view)
            if W // 3 < center_x < 2 * W // 3:
                if avg_depth > max_risk_score:
                    max_risk_score = avg_depth
                    most_critical_depth_change = depth_change

        # Cleanup old tracks
        for tid in list(track_history.keys()):
            if tid not in current_tracked_ids:
                del track_history[tid]

        # 5. Pathfinding (A*)
        start_node = (GRID_ROWS - 1, GRID_COLS // 2)
        goal_node = (0, GRID_COLS // 2)
        grid[start_node] = 0
        grid[goal_node] = 0

        path = astar(grid, start_node, goal_node)

        # 6. Driving Commands logic with Rel Velocity
        # Decouple Speed and Steering to allow following a car even if the path is blocked
        
        # 6.1 Speed Control
        if max_risk_score > 200:
            speed_cmd = "STOP"
        elif max_risk_score > 150:
            if most_critical_depth_change > 1.5:
                speed_cmd = "STOP"
            elif most_critical_depth_change < -1.5:
                speed_cmd = "MAINTAIN_SPEED"
            else:
                speed_cmd = "SLOW_DOWN"
        elif max_risk_score > 80:
            if most_critical_depth_change > 2.0:
                speed_cmd = "SLOW_DOWN"
            elif most_critical_depth_change < -1.5:
                speed_cmd = "ACCELERATE"
            else:
                speed_cmd = "MAINTAIN_SPEED"
        else:
            speed_cmd = "ACCELERATE"

        # 6.2 Steering Control
        steering_cmd = None
        if path and len(path) > 5:
            ahead_node = path[min(5, len(path)-1)]
            dx = ahead_node[1] - start_node[1]
            if dx < -2:
                steering_cmd = "TURN_LEFT"
            elif dx > 2:
                steering_cmd = "TURN_RIGHT"

        # 6.3 Combine
        if speed_cmd == "STOP":
            command = "STOP"
        elif steering_cmd:
            command = steering_cmd
        else:
            command = speed_cmd

        # Minimap Visualization
        minimap = cv2.resize((1-grid)*255, (W//3, H//3), interpolation=cv2.INTER_NEAREST)
        minimap = cv2.cvtColor(minimap, cv2.COLOR_GRAY2BGR)
        if path:
            for (r, c) in path:
                mx = int((c / GRID_COLS) * (W // 3))
                my = int((r / GRID_ROWS) * (H // 3))
                cv2.circle(minimap, (mx, my), 2, (0, 255, 0), -1)
        
        # Draw HUD
        annotated_frame[0:H//3, W - W//3:W] = minimap
        risk_color = (0, 255, 0)
        if max_risk_score > 160: risk_color = (0, 165, 255) # Orange
        if max_risk_score > 220: risk_color = (0, 0, 255)   # Red
        
        cv2.putText(annotated_frame, f"RISK SCORE: {max_risk_score:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
        cv2.putText(annotated_frame, f"CMD: {command}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

        # Show Output
        combined_view = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow('Autonomous Dashcam Analysis', combined_view)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Dashcam Analysis on a Video File")
    parser.add_argument("--video", type=str, default=r"C:\Allen\ml projects\autonomus path\example video\test_video.mp4", help="Path to the dashcam video file")
    args = parser.parse_args()
    main(args.video)
