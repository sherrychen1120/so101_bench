import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Union
import logging


class TaskProgressLabeler:
    """Interactive video player for manual progress stage labeling with dual video support."""
    
    def __init__(
        self, 
        video_paths: List[Path], 
        progress_stage_labels: list[str], 
        fps: float = 30.0, 
        horizon_s: float = None,
        title: str = "",
    ):
        self.video_paths = video_paths
        self.caps = {}
        self.total_frames = 0
        self.fps = fps
        self.horizon_s = horizon_s
        self.title = title
        
        # Initialize video captures
        for i, video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            video_name = video_path.stem
            self.caps[video_name] = cap
            
            # Use the first video to determine timing
            if i == 0:
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.duration_s = self.total_frames / fps
        self.current_frame = 0
        self.is_playing = False
        
        # Progress stage labels
        self.progress_stage_labels = progress_stage_labels
        
        # Store labeled time points: [(time, stage), ...] sorted by time
        self.labeled_points = []  # List of (time, stage) tuples
        
        # For point selection and deletion
        self.selected_point_index = None  # Index into labeled_points list
        
        # Colors for visualization (BGR format for OpenCV)
        self.stage_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        self.error_color = (0, 0, 255)  # Red for overlapping intervals
        self.selected_color = (255, 255, 255)  # White for selected interval
        
        print(f"Videos loaded: {[p.name for p in video_paths]}")
        print(f"Duration: {self.duration_s:.2f}s, Frames: {self.total_frames}, FPS: {fps}")
        if self.horizon_s:
            print(f"Labeling horizon: {self.horizon_s:.2f}s (labeling only allowed up to this point)")
        print("\nControls:")
        print("  SPACE - Play/Pause")
        print("  LEFT/RIGHT - Seek backward/forward (1 second)")
        print("  UP/DOWN - Seek backward/forward (5 seconds)")
        print(f"  0-{len(self.progress_stage_labels) - 1} - Label current time point with progress stage (0={self.progress_stage_labels[0]}, 1={self.progress_stage_labels[1]}, etc.)")
        print("  S - Show current labels")
        print("  Q - Quit and save")
        print("  R - Reset all labels")
        print("  TAB - Select next point for deletion")
        print("  X/BACKSPACE - Delete selected point")
        print("  Ctrl+C - Gracefully quit and save")
    
    def get_current_time(self) -> float:
        """Get current playback time in seconds."""
        # TODO(sherry): Use sync_log to map video frames to timestamps.
        return self.current_frame / self.fps
    
    def seek_to_time(self, time_s: float) -> None:
        """Seek to specific time in seconds."""
        frame = int(time_s * self.fps)
        frame = max(0, min(frame, self.total_frames - 1))
        self.current_frame = frame
        
        # Seek all video captures
        for cap in self.caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    def seek_relative(self, delta_s: float) -> None:
        """Seek relative to current position."""
        new_time = self.get_current_time() + delta_s
        self.seek_to_time(new_time)
    
    def label_current_time(self, stage_idx: int) -> None:
        """Label the current time point with a progress stage."""
        if 0 <= stage_idx < len(self.progress_stage_labels):
            current_time = self.get_current_time()
            
            # Check horizon constraint
            if self.horizon_s and current_time > self.horizon_s:
                print(f"Warning: Cannot label beyond horizon ({self.horizon_s:.2f}s). Current time: {current_time:.2f}s")
                return
            
            stage = self.progress_stage_labels[stage_idx]
            
            self.labeled_points.append((current_time, stage))
            # Keep points sorted by time
            self.labeled_points.sort(key=lambda x: x[0])
            print(f"Labeled time point {current_time:.2f}s as '{stage}'")
    
    def get_intervals_from_points(self) -> dict[str, list[tuple[float, float]]]:
        """Generate intervals from labeled time points.
        
        The interval (start, end) is labeled with the stage that ENDS at 'end' time.
        If multiple labels are at the same time, earlier labels (by task_spec order) 
        get the interval, later labels get zero-duration intervals.
        
        Returns:
            Dict mapping stage names to list of intervals (start_time, end_time).
        """
        if not self.labeled_points:
            return {}
        
        intervals = defaultdict(list)
        
        # Sort points by time, then by stage order for simultaneous points
        def sort_key(point):
            time, stage = point
            stage_order = self.progress_stage_labels.index(stage) if stage in self.progress_stage_labels else 999
            return (time, stage_order)
        
        sorted_points = sorted(self.labeled_points, key=sort_key)
        
        # Group points by time to handle simultaneous labels
        time_groups = []
        current_time = None
        current_group = []
        
        for time, stage in sorted_points:
            if current_time is None or abs(time - current_time) < 0.01:  # Same time (within 10ms)
                if current_time is None:
                    current_time = time
                current_group.append((time, stage))
            else:
                # New time group
                if current_group:
                    time_groups.append((current_time, current_group))
                current_time = time
                current_group = [(time, stage)]
        
        # Add the last group
        if current_group:
            time_groups.append((current_time, current_group))
        
        # Generate intervals
        prev_time = 0.0
        
        for group_time, group_stages in time_groups:
            # The first stage in the group (earliest in task_spec order) gets the interval
            if group_time > prev_time:
                first_stage = group_stages[0][1]  # Take the stage from the first (earliest order) point
                intervals[first_stage].append((prev_time, group_time))
            
            # Any additional stages at the same time get zero-duration intervals
            for i in range(1, len(group_stages)):
                stage = group_stages[i][1]
                intervals[stage].append((group_time, group_time))
            
            prev_time = group_time
        
        return dict(intervals)
    
    def show_labels(self) -> None:
        """Display current labels."""
        print("\nCurrent time points:")
        if not self.labeled_points:
            print("  No points labeled yet.")
        else:
            for time, stage in sorted(self.labeled_points, key=lambda x: x[0]):
                print(f"  {time:.2f}s: {stage}")
        
        print("\nDerived intervals:")
        intervals = self.get_intervals_from_points()
        if not intervals:
            print("  No intervals yet.")
        else:
            for stage, stage_intervals in intervals.items():
                print(f"  {stage}:")
                for start, end in stage_intervals:
                    print(f"    {start:.2f}s - {end:.2f}s")
    
    def reset_labels(self) -> None:
        """Reset all labels."""
        self.labeled_points.clear()
        self.selected_point_index = None
        print("All labels reset.")
    
    def is_valid(self) -> bool:
        """Check if current labels are valid. Points can't overlap by definition."""
        return True  # Time points can't overlap, so always valid
    
    def select_next_point(self) -> None:
        """Select the next point for deletion."""
        if not self.labeled_points:
            print("No points to select.")
            return
        
        if self.selected_point_index is None:
            # Select first point
            self.selected_point_index = 0
        else:
            # Move to next point (cycle through)
            self.selected_point_index = (self.selected_point_index + 1) % len(self.labeled_points)
        
        time, stage = self.labeled_points[self.selected_point_index]
        print(f"Selected: point {self.selected_point_index} ({time:.2f}s: {stage})")
    
    def delete_selected_point(self) -> None:
        """Delete the currently selected point."""
        if self.selected_point_index is None:
            print("No point selected. Use TAB to select a point first.")
            return
        
        if self.selected_point_index >= len(self.labeled_points):
            print("Selected point no longer exists.")
            self.selected_point_index = None
            return
        
        point = self.labeled_points[self.selected_point_index]
        self.labeled_points.pop(self.selected_point_index)
        
        print(f"Deleted point ({point[0]:.2f}s: {point[1]})")
        
        # Adjust selected index if needed
        if self.selected_point_index >= len(self.labeled_points):
            self.selected_point_index = None if not self.labeled_points else len(self.labeled_points) - 1
    
    def create_combined_frame(self, frames: List[np.ndarray]) -> np.ndarray:
        """Combine multiple video frames into a single display frame."""
        if len(frames) == 1:
            # Single video - make it larger
            frame = frames[0]
            height, width = frame.shape[:2]
            # Scale up by 1.5x
            new_width, new_height = int(width * 1.5), int(height * 1.5)
            return cv2.resize(frame, (new_width, new_height))
        
        elif len(frames) == 2:
            # Dual video - side by side, larger size
            frame1, frame2 = frames
            
            # Resize each frame to be larger
            height, width = frame1.shape[:2]
            new_width, new_height = int(width * 1.2), int(height * 1.2)
            
            frame1_resized = cv2.resize(frame1, (new_width, new_height))
            frame2_resized = cv2.resize(frame2, (new_width, new_height))
            
            # Combine side by side
            combined = np.hstack([frame1_resized, frame2_resized])
            return combined
        
        else:
            # More than 2 videos - create a grid
            # For now, just use the first two
            logging.warning("Only support up to 2 videos for now.")
            return self.create_combined_frame(frames[:2])

    def add_overlay_text(self, frame: np.ndarray) -> np.ndarray:
        """Add overlay information to the frame."""
        current_time = self.get_current_time()
        
        # Main time display
        overlay_text = f"Time: {current_time:.2f}s / {self.duration_s:.2f}s"
        
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Selected point information
        if self.selected_point_index is not None and self.selected_point_index < len(self.labeled_points):
            point_time, point_stage = self.labeled_points[self.selected_point_index]
            selected_text = f"Selected: point {self.selected_point_index} ({point_time:.2f}s: {point_stage})"
            cv2.putText(frame, selected_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2, cv2.LINE_AA)  # Yellow text for visibility
        
        # Horizon information
        if self.horizon_s:
            if current_time > self.horizon_s:
                horizon_text = f"BEYOND HORIZON ({self.horizon_s:.2f}s) - NO LABELING"
                cv2.putText(frame, horizon_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2, cv2.LINE_AA)  # Red text for warning
            else:
                horizon_text = f"Horizon: {self.horizon_s:.2f}s"
                cv2.putText(frame, horizon_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)  # Green text for OK
        
        # Video labels
        if len(self.caps) > 1:
            # Label each video
            frame_height = frame.shape[0]
            frame_width = frame.shape[1] // len(self.caps)
            
            for i, video_path in enumerate(self.video_paths):
                video_name = video_path.stem.replace('video_', '').upper()
                x_pos = i * frame_width + 10
                cv2.putText(frame, video_name, (x_pos, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        return frame

    def create_visualization_panel(self, width: int, height: int) -> np.ndarray:
        """Create a visualization panel showing all labeled time points and derived intervals."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not self.labeled_points:
            # No points to show
            cv2.putText(panel, "No points labeled yet", (10, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return panel
        
        # Calculate layout
        stage_height = height // (len(self.progress_stage_labels) + 1)  # +1 for status bar
        time_scale = width / self.duration_s if self.duration_s > 0 else 1
        
        # Get derived intervals for visualization
        intervals = self.get_intervals_from_points()
        
        # Draw time axis
        cv2.line(panel, (50, height - 30), (width - 10, height - 30), (255, 255, 255), 1)
        
        # Draw time markers
        for i in range(0, int(self.duration_s) + 1, max(1, int(self.duration_s / 10))):
            x = int(50 + i * time_scale)
            if x < width - 10:
                cv2.line(panel, (x, height - 35), (x, height - 25), (255, 255, 255), 1)
                cv2.putText(panel, f"{i}s", (x - 10, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw stages and intervals
        for stage_idx, stage in enumerate(self.progress_stage_labels):
            y = stage_height * (stage_idx + 1)
            
            # Draw stage label
            cv2.putText(panel, stage, (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Draw stage baseline
            cv2.line(panel, (50, y), (width - 10, y), (128, 128, 128), 1)
            
            # Draw intervals for this stage
            if stage in intervals:
                color_idx = stage_idx % len(self.stage_colors)
                stage_color = self.stage_colors[color_idx]
                
                for start, end in intervals[stage]:
                    start_x = int(50 + start * time_scale)
                    end_x = int(50 + end * time_scale)
                    
                    # Draw interval line
                    cv2.line(panel, (start_x, y - 10), (end_x, y - 10), stage_color, 3)
                    
                    # Draw interval markers
                    cv2.circle(panel, (start_x, y - 10), 2, stage_color, -1)
                    cv2.circle(panel, (end_x, y - 10), 2, stage_color, -1)
        
        # Draw time points as vertical lines
        for point_idx, (time, stage) in enumerate(self.labeled_points):
            x = int(50 + time * time_scale)
            if 50 <= x <= width - 10:
                # Find stage color
                stage_idx = self.progress_stage_labels.index(stage) if stage in self.progress_stage_labels else 0
                color_idx = stage_idx % len(self.stage_colors)
                point_color = self.stage_colors[color_idx]
                
                # Check if this point is selected
                if self.selected_point_index == point_idx:
                    point_color = self.selected_color
                    thickness = 3
                else:
                    thickness = 2
                
                # Draw vertical line for the time point
                cv2.line(panel, (x, 10), (x, height - 50), point_color, thickness)
                
                # Draw point marker
                cv2.circle(panel, (x, 15), 4, point_color, -1)
                
                # Draw stage label near the point
                cv2.putText(panel, stage[:3], (x - 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.3, point_color, 1)
        
        # Draw current time indicator
        current_time = self.get_current_time()
        current_x = int(50 + current_time * time_scale)
        if 50 <= current_x <= width - 10:
            cv2.line(panel, (current_x, 10), (current_x, height - 40), (255, 255, 0), 2)
            cv2.putText(panel, f"Now: {current_time:.1f}s", (current_x + 5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw status indicator
        status_text = f"Points: {len(self.labeled_points)}"
        status_color = (0, 255, 0)
        cv2.putText(panel, f"Status: {status_text}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return panel

    def play(self) -> dict[str, list[tuple[float, float]]]:
        """Start interactive video player and return labeled intervals."""
        window_name = f"{self.title}" if self.title else "Video Player - Manual Labeling"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Make window larger to accommodate visualization panel
        cv2.resizeWindow(window_name, 1600, 1000)
        
        clock = cv2.getTickCount()
        
        try:
            while True:
                # Read frames from all videos
                frames = []
                all_frames_valid = True
                
                for video_name, cap in self.caps.items():
                    # Always seek to the current frame to ensure synchronization
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = cap.read()
                    
                    if not ret:
                        all_frames_valid = False
                        break
                    frames.append(frame)
                
                if not all_frames_valid:
                    # End of video - pause instead of looping
                    self.is_playing = False
                    print(f"Reached end of video at {self.get_current_time():.2f}s")
                    # Reset to last valid frame
                    if self.current_frame > 0:
                        self.current_frame -= 1
                        self.seek_to_time(self.get_current_time())
                    continue
                
                # Create combined display frame
                combined_frame = self.create_combined_frame(frames)
                
                # Add overlay information
                video_frame = self.add_overlay_text(combined_frame)
                
                # Create visualization panel
                viz_panel = self.create_visualization_panel(video_frame.shape[1], 200)
                
                # Combine video and visualization vertically
                display_frame = np.vstack([video_frame, viz_panel])
                
                cv2.imshow(window_name, display_frame)
                
                # Check if window was closed externally
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow was closed. Exiting labeler...")
                    break
                
                # Handle keyboard input
                key = cv2.waitKey(1 if self.is_playing else 0) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    break
                elif key == ord(' '):
                    # Play/pause
                    self.is_playing = not self.is_playing
                    print("Playing" if self.is_playing else "Paused")
                elif key == 81 or key == 2:  # Left arrow
                    self.seek_relative(-1.0)
                    self.is_playing = False
                elif key == 83 or key == 3:  # Right arrow
                    self.seek_relative(1.0)
                    self.is_playing = False
                elif key == 82 or key == 0:  # Up arrow
                    self.seek_relative(-5.0)
                    self.is_playing = False
                elif key == 84 or key == 1:  # Down arrow
                    self.seek_relative(5.0)
                    self.is_playing = False
                elif key in [ord(str(i)) for i in range(len(self.progress_stage_labels))]:
                    # Label current time point with stage
                    stage_idx = int(chr(key))
                    self.label_current_time(stage_idx)
                elif key == ord('s'):
                    self.show_labels()
                elif key == ord('r'):
                    self.reset_labels()
                elif key == 9:  # Tab
                    self.select_next_point()
                elif key == ord('x'):  # Use 'x' as reliable delete key
                    self.delete_selected_point()
                elif key == 255 and not self.is_playing:
                    # Only treat 255 as delete when paused AND a point is selected
                    print("Detected delete key (255) when video is paused. Deleting selected point")
                    self.delete_selected_point()
                elif key == 8:  # Backspace as alternative
                    self.delete_selected_point()
                
                # Auto advance if playing
                if self.is_playing:
                    new_tick = cv2.getTickCount()
                    elapsed = (new_tick - clock) / cv2.getTickFrequency()
                    if elapsed >= 1.0 / self.fps:
                        self.current_frame += 1
                        # Don't loop - just stop at the end
                        if self.current_frame >= self.total_frames:
                            self.current_frame = self.total_frames - 1
                            self.is_playing = False
                            print("Reached end of video")
                        clock = new_tick
        
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Saving labels and exiting gracefully...")
        
        cv2.destroyAllWindows()
        for cap in self.caps.values():
            cap.release()
        
        # Generate intervals from labeled points
        result = {}
        intervals = self.get_intervals_from_points()
        
        print(f"\n✓ Generated {sum(len(stage_intervals) for stage_intervals in intervals.values())} intervals from {len(self.labeled_points)} time points")
        
        for stage, stage_intervals in intervals.items():
            result[stage] = [[start, end] for start, end in stage_intervals]
        
        return result
