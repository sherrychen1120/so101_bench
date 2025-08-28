import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Union
import logging


class TaskProgressLabeler:
    """Interactive video player for manual progress stage labeling with dual video support."""
    
    def __init__(self, video_paths: Union[str, List[str]], progress_stage_labels: list[str], fps: float = 30.0, horizon_s: float = None):
        # Handle both single video and multiple videos
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        
        self.video_paths = video_paths
        self.caps = []
        self.total_frames = 0
        self.fps = fps
        self.horizon_s = horizon_s
        
        # Initialize video captures
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            self.caps.append(cap)
            
            # Use the first video to determine timing
            if len(self.caps) == 1:
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
        
        print(f"Videos loaded: {[Path(p).name for p in video_paths]}")
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
        for cap in self.caps:
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
        print("All labels reset.")
    
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
                video_name = Path(video_path).stem.replace('video_', '').upper()
                x_pos = i * frame_width + 10
                cv2.putText(frame, video_name, (x_pos, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        return frame

    def play(self) -> dict[str, list[tuple[float, float]]]:
        """Start interactive video player and return labeled intervals."""
        window_name = "Video Player - Manual Labeling"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Make window larger
        cv2.resizeWindow(window_name, 1600, 800)
        
        clock = cv2.getTickCount()
        
        while True:
            # Read frames from all videos
            frames = []
            all_frames_valid = True
            
            for cap in self.caps:
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
            display_frame = self.add_overlay_text(combined_frame)
            
            cv2.imshow(window_name, display_frame)
            
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
        
        cv2.destroyAllWindows()
        for cap in self.caps:
            cap.release()
        
        # Convert to the expected format
        result = {}
        for stage, intervals in self.labeled_intervals.items():
            result[stage] = [[start, end] for start, end in intervals]
        
        return result

