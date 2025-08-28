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
        
        # Store labeled intervals: {stage: [(start_time, end_time), ...]}
        self.labeled_intervals = defaultdict(list)
        self.current_labeling_stage = None
        self.labeling_start_time = None
        
        # For interval selection and deletion
        self.selected_interval = None  # (stage, interval_index)
        
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
        print(f"  0-{len(self.progress_stage_labels) - 1} - Start labeling progress stage (0={self.progress_stage_labels[0]}, 1={self.progress_stage_labels[1]}, etc.)")
        print("  ENTER - End current labeling")
        print("  S - Show current labels")
        print("  Q - Quit and save")
        print("  R - Reset all labels")
        print("  TAB - Select next interval for deletion")
        print("  X/BACKSPACE - Delete selected interval")
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
    
    def start_labeling(self, stage_idx: int) -> None:
        """Start labeling a progress stage."""
        if 0 <= stage_idx < len(self.progress_stage_labels):
            if self.current_labeling_stage is not None:
                print(f"Warning: Already labeling {self.current_labeling_stage}. End current labeling first.")
                return
            
            current_time = self.get_current_time()
            # Check horizon constraint
            if self.horizon_s and current_time > self.horizon_s:
                print(f"Warning: Cannot start labeling beyond horizon ({self.horizon_s:.2f}s). Current time: {current_time:.2f}s")
                return
            
            self.current_labeling_stage = self.progress_stage_labels[stage_idx]
            self.labeling_start_time = current_time
            print(f"Started labeling '{self.current_labeling_stage}' from {self.labeling_start_time:.2f}s")
    
    def end_labeling(self) -> None:
        """End current labeling and save interval."""
        if self.current_labeling_stage is None:
            print("No active labeling to end.")
            return
        
        end_time = self.get_current_time()
        
        # Clip end time to horizon if needed
        if self.horizon_s and end_time > self.horizon_s:
            end_time = self.horizon_s
            print(f"Warning: End time clipped to horizon ({self.horizon_s:.2f}s)")
        
        if end_time > self.labeling_start_time:
            interval = (self.labeling_start_time, end_time)
            self.labeled_intervals[self.current_labeling_stage].append(interval)
            print(f"Labeled '{self.current_labeling_stage}': {self.labeling_start_time:.2f}s - {end_time:.2f}s")
        else:
            print("Invalid interval: end time must be after start time.")
        
        self.current_labeling_stage = None
        self.labeling_start_time = None
    
    def show_labels(self) -> None:
        """Display current labels."""
        print("\nCurrent labels:")
        if not self.labeled_intervals:
            print("  No labels yet.")
        else:
            for stage, intervals in self.labeled_intervals.items():
                print(f"  {stage}:")
                for start, end in intervals:
                    print(f"    {start:.2f}s - {end:.2f}s")
        
        if self.current_labeling_stage:
            print(f"\nCurrently labeling: {self.current_labeling_stage} (started at {self.labeling_start_time:.2f}s)")
    
    def reset_labels(self) -> None:
        """Reset all labels."""
        self.labeled_intervals.clear()
        self.current_labeling_stage = None
        self.labeling_start_time = None
        self.selected_interval = None
        print("All labels reset.")
    
    def detect_overlaps(self) -> dict[str, list[tuple[str, int, int]]]:
        """Detect overlapping intervals between different stages.
        
        Returns:
            Dict mapping stage names to list of overlapping intervals.
            Each overlap is (other_stage, self_interval_idx, other_interval_idx)
        """
        overlaps = defaultdict(list)
        
        stages = list(self.labeled_intervals.keys())
        for i, stage1 in enumerate(stages):
            for j, stage2 in enumerate(stages[i+1:], i+1):
                for idx1, (start1, end1) in enumerate(self.labeled_intervals[stage1]):
                    for idx2, (start2, end2) in enumerate(self.labeled_intervals[stage2]):
                        # Check if intervals overlap
                        if start1 < end2 and start2 < end1:
                            overlaps[stage1].append((stage2, idx1, idx2))
                            overlaps[stage2].append((stage1, idx2, idx1))
        
        return overlaps
    
    def is_valid(self) -> bool:
        """Check if current labels are valid (no overlaps)."""
        return len(self.detect_overlaps()) == 0
    
    def select_next_interval(self) -> None:
        """Select the next interval for deletion."""
        if not self.labeled_intervals:
            print("No intervals to select.")
            return
        
        # Get all intervals in order
        all_intervals = []
        for stage in self.progress_stage_labels:
            if stage in self.labeled_intervals:
                for idx, interval in enumerate(self.labeled_intervals[stage]):
                    all_intervals.append((stage, idx))
        
        if not all_intervals:
            print("No intervals to select.")
            return
        
        if self.selected_interval is None:
            # Select first interval
            self.selected_interval = all_intervals[0]
        else:
            # Find current selection and move to next
            try:
                current_idx = all_intervals.index(self.selected_interval)
                next_idx = (current_idx + 1) % len(all_intervals)
                self.selected_interval = all_intervals[next_idx]
            except ValueError:
                # Current selection no longer exists, select first
                self.selected_interval = all_intervals[0]
        
        stage, idx = self.selected_interval
        start, end = self.labeled_intervals[stage][idx]
        print(f"Selected: {stage} interval {idx} ({start:.2f}s - {end:.2f}s)")
    
    def delete_selected_interval(self) -> None:
        """Delete the currently selected interval."""
        if self.selected_interval is None:
            print("No interval selected. Use TAB to select an interval first.")
            return
        
        stage, idx = self.selected_interval
        if stage not in self.labeled_intervals or idx >= len(self.labeled_intervals[stage]):
            print("Selected interval no longer exists.")
            self.selected_interval = None
            return
        
        interval = self.labeled_intervals[stage][idx]
        self.labeled_intervals[stage].pop(idx)
        
        # Clean up empty stages
        if not self.labeled_intervals[stage]:
            del self.labeled_intervals[stage]
        
        print(f"Deleted {stage} interval ({interval[0]:.2f}s - {interval[1]:.2f}s)")
        self.selected_interval = None
    
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
        if self.current_labeling_stage:
            overlay_text += f" | Labeling: {self.current_labeling_stage}"
        
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
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
        """Create a visualization panel showing all labeled intervals."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not self.labeled_intervals:
            # No intervals to show
            cv2.putText(panel, "No intervals labeled yet", (10, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return panel
        
        # Calculate layout
        stage_height = height // (len(self.progress_stage_labels) + 1)  # +1 for status bar
        time_scale = width / self.duration_s if self.duration_s > 0 else 1
        
        # Get overlaps for error highlighting
        overlaps = self.detect_overlaps()
        
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
            
            if stage in self.labeled_intervals:
                color_idx = stage_idx % len(self.stage_colors)
                stage_color = self.stage_colors[color_idx]
                
                for interval_idx, (start, end) in enumerate(self.labeled_intervals[stage]):
                    start_x = int(50 + start * time_scale)
                    end_x = int(50 + end * time_scale)
                    
                    # Determine color (normal, error, or selected)
                    color = stage_color
                    thickness = 3
                    
                    # Check if this interval is selected
                    if (self.selected_interval and 
                        self.selected_interval[0] == stage and 
                        self.selected_interval[1] == interval_idx):
                        color = self.selected_color
                        thickness = 5
                    # Check if this interval has overlaps
                    elif stage in overlaps:
                        for other_stage, self_idx, other_idx in overlaps[stage]:
                            if self_idx == interval_idx:
                                color = self.error_color
                                thickness = 4
                                break
                    
                    # Draw interval line
                    cv2.line(panel, (start_x, y - 10), (end_x, y - 10), color, thickness)
                    
                    # Draw interval markers
                    cv2.circle(panel, (start_x, y - 10), 3, color, -1)
                    cv2.circle(panel, (end_x, y - 10), 3, color, -1)
        
        # Draw current time indicator
        current_time = self.get_current_time()
        current_x = int(50 + current_time * time_scale)
        if 50 <= current_x <= width - 10:
            cv2.line(panel, (current_x, 10), (current_x, height - 40), (255, 255, 0), 2)
            cv2.putText(panel, f"Now: {current_time:.1f}s", (current_x + 5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw status indicator
        is_valid = self.is_valid()
        status_text = "VALID" if is_valid else "INVALID (Overlaps detected)"
        status_color = (0, 255, 0) if is_valid else (0, 0, 255)
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
                    # Start labeling stage
                    stage_idx = int(chr(key))
                    self.start_labeling(stage_idx)
                elif key == 13:  # Enter
                    self.end_labeling()
                elif key == ord('s'):
                    self.show_labels()
                elif key == ord('r'):
                    self.reset_labels()
                elif key == 9:  # Tab
                    self.select_next_interval()
                elif key == ord('x'):  # Use 'x' as reliable delete key
                    self.delete_selected_interval()
                elif key == 255 and not self.is_playing:
                    # Only treat 255 as delete when paused AND an interval is selected
                    print("Detected delete key (255) when video is paused. Deleting selected interval")
                    self.delete_selected_interval()
                elif key == 8:  # Backspace as alternative
                    self.delete_selected_interval()
                
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
        
        # Validate final labels
        result = {}
        if not self.is_valid():
            print("\n" + "="*50)
            print("WARNING: Labels contain overlapping intervals! Discarding these labels.")
            overlaps = self.detect_overlaps()
            for stage, overlap_list in overlaps.items():
                print(f"Stage '{stage}' has overlaps:")
                for other_stage, self_idx, other_idx in overlap_list:
                    self_interval = self.labeled_intervals[stage][self_idx]
                    other_interval = self.labeled_intervals[other_stage][other_idx]
                    print(f"  - {stage} interval {self_idx} ({self_interval[0]:.2f}s-{self_interval[1]:.2f}s) "
                          f"overlaps with {other_stage} interval {other_idx} ({other_interval[0]:.2f}s-{other_interval[1]:.2f}s)")
            print("="*50)
            return result
        else:
            print("\n✓ All labels are valid (no overlaps detected)")

            for stage, intervals in self.labeled_intervals.items():
                result[stage] = [[start, end] for start, end in intervals]
            return result
