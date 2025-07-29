import gymnasium as gym
import numpy as np
import cv2
import ale_py

def main():
    env = gym.make("ALE/Seaquest-v5", render_mode="rgb_array", frameskip=1)
    obs, info = env.reset()
    prev_ram = env.unwrapped.ale.getRAM().copy()

    font = cv2.FONT_HERSHEY_PLAIN

    # Track potential candidates for specific game elements
    oxygen_candidate = 102  # You mentioned byte 1, but let's also track others
    score_candidates = []
    position_candidates = []

    print("[INFO] Press any key to step, 'q' to quit, 's' to save current RAM state")
    print("[INFO] Suggested testing: Move submarine, rescue divers, shoot enemies, watch oxygen\n")

    frame_count = 0

    while True:
        # Try different actions to trigger different RAM changes
        # 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN, 6=UPRIGHT, 7=UPLEFT, 8=DOWNRIGHT, 9=DOWNLEFT
        action = 5  # Default to no-op, but you can change this

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ram = env.unwrapped.ale.getRAM()

        frame_count += 1

        # === Detect ALL changes (not just increases) ===
        changed = [(i, int(prev_ram[i]), int(ram[i])) for i in range(128) if ram[i] != prev_ram[i]]
        increasing = [(i, old, new) for i, old, new in changed if new > old]
        decreasing = [(i, old, new) for i, old, new in changed if new < old]

        # === Display current key values ===
        if frame_count % 30 == 0 or changed:  # Show every 30 frames or when changes occur
            print(f"\n=== Frame {frame_count} ===")
            print(f"Score from info: {info.get('score', 'N/A')}")
            print(f"Lives: {info.get('lives', 'N/A')}")

            # Show confirmed and suspected values
            print(f"RAM[102] (oxygen): {ram[102]}")
            print(f"RAM[1] (frame timer): {ram[1]}")

            # Show diver-related candidates
            print(f"Diver candidates - RAM[79]: {ram[79]} | RAM[80]: {ram[80]} | RAM[93]: {ram[93]}")
            print(f"Diver sync group - RAM[31]: {ram[31]} | RAM[32]: {ram[32]} | RAM[33]: {ram[33]}")
            print(f"Diver timer? - RAM[18]: {ram[18]}")

            if increasing:
                print(f"[INCREASING] {increasing[:5]}")  # Show first 5
            if decreasing:
                print(f"[DECREASING] {decreasing[:5]}")  # Show first 5

        # === Look for specific patterns ===
        # Score typically increases in specific increments (10, 20, 50, etc.)
        if reward > 0:
            print(f"[REWARD] Got reward {reward}! Check these changes: {changed}")

        # Oxygen typically decreases over time
        if ram[102] < prev_ram[102]:
            print(f"[OXYGEN?] RAM[102] decreased: {prev_ram[102]} → {ram[102]}")

        prev_ram = ram.copy()

        # === Create visual overlay ===
        overlay = obs.copy()
        y_offset = 20

        # Show current suspected values
        cv2.putText(overlay, f"Oxygen?: {ram[102]}", (10, y_offset), font, 1.0, (255, 255, 0), 1)
        cv2.putText(overlay, f"Score: {info.get('score', 'N/A')}", (10, y_offset + 18), font, 1.0, (255, 255, 0), 1)
        cv2.putText(overlay, f"Lives: {info.get('lives', 'N/A')}", (10, y_offset + 36), font, 1.0, (255, 255, 0), 1)

        # Show recent changes
        for j, (i, old_val, new_val) in enumerate(changed[:8]):  # show up to 8
            color = (0, 255, 0) if new_val > old_val else (0, 0, 255)  # Green for increase, red for decrease
            txt = f"RAM[{i}] {old_val}→{new_val}"
            cv2.putText(overlay, txt, (200, y_offset + j * 18), font, 1.0, color, 1)

        # === Resize and display ===
        resized = cv2.resize(overlay, (obs.shape[1] * 3, obs.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Seaquest RAM Mapper", resized)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # Save current RAM state
            print(f"\n=== SAVED RAM STATE (Frame {frame_count}) ===")
            for i in range(0, 128, 8):
                row = " ".join(f"{ram[j]:3d}" for j in range(i, min(i+8, 128)))
                print(f"RAM[{i:2d}-{min(i+7, 127):2d}]: {row}")
            print()

        if done:
            print(f"[GAME OVER] Final score: {info.get('score', 'N/A')}")
            obs, info = env.reset()
            frame_count = 0

    env.close()
    cv2.destroyAllWindows()

# Additional helper function to analyze specific RAM patterns
def analyze_ram_patterns():
    """
    Run this separately to look for common Atari patterns
    """
    print("=== Common Atari 2600 RAM Patterns ===")
    print("Score: Often stored in BCD format across multiple bytes")
    print("Position: Usually 1-2 bytes for X/Y coordinates")
    print("Lives: Single byte, often decrements")
    print("Timers: Decrease each frame or every few frames")
    print("Object states: Bit flags or small integers")
    print("\nFor Seaquest specifically:")
    print("- Oxygen: Should decrease over time, increase at surface")
    print("- Submarine position: Changes with movement")
    print("- Divers: Position and rescue state")
    print("- Depth: Current level/depth indicator")

if __name__ == "__main__":
    main()
    # Uncomment to see pattern analysis:
    analyze_ram_patterns()