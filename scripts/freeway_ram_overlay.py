import gymnasium as gym
import numpy as np
import cv2
import random
import csv
import ale_py

# --- Helper function for address conversion ---
STELLA_RAM_BASE_ADDR = 0x80 # The starting address of Atari 2600 RAM in Stella's hex view

def stella_addr_to_python_idx(stella_hex_addr):
    """Converts a Stella debugger hex RAM address (0x80-0xFF) to a Python ram array index (0-127)."""
    if not (0x80 <= stella_hex_addr <= 0xFF):
        print(f"Warning: Stella hex address 0x{stella_hex_addr:02x} is outside 0x80-0xFF RAM range. Returning -1.")
        return -1
    return stella_hex_addr - STELLA_RAM_BASE_ADDR

# --- Main script ---
def main():
    env = gym.make("ALE/Freeway-v5", render_mode="rgb_array", frameskip=1)
    obs, info = env.reset()
    prev_ram = env.unwrapped.ale.getRAM().copy()

    font = cv2.FONT_HERSHEY_PLAIN

    # --- Confirmed/Candidate RAM Addresses (using Python indices) ---
    # Python index 14 is Stella 0x8E. Confirm this with manual Stella debugger observation.
    player_chicken_y_candidate_py_idx = stella_addr_to_python_idx(0x8E)

    # Score candidates (common guesses for 2-byte BCD score)
    score_candidate_1_py_idx = stella_addr_to_python_idx(0xE4)
    score_candidate_2_py_idx = stella_addr_to_python_idx(0xE5)
    general_timer_py_idx = stella_addr_to_python_idx(0x81)

    # --- Car Position Candidates (Python indices) - Now in 4-byte blocks ---
    # These are illustrative. You'll need to confirm what each byte means.
    car_blocks_display = {
        "Car 1 Block (0xA1-0xA4)": [stella_addr_to_python_idx(i) for i in range(0xA1, 0xA5)], # RAM[33]-RAM[36]
        "Car 2 Block (0xA7-0xAA)": [stella_addr_to_python_idx(i) for i in range(0xA7, 0xAB)], # RAM[39]-RAM[42]
        "Car 3 Block (0xB0-0xB3)": [stella_addr_to_python_idx(i) for i in range(0xB0, 0xB4)], # RAM[48]-RAM[51] (Includes your previous strong candidates 48,49,50!)
        "Car 4 Block (0xB4-0xB7)": [stella_addr_to_python_idx(i) for i in range(0xB4, 0xB8)], # RAM[52]-RAM[55]
    }

    # For full CSV logging, you might want a wider range of potential car indices.
    all_car_indices_for_log = [stella_addr_to_python_idx(i) for i in range(0x90, 0xD0)] # Covers 0x90-0xCF

    print(f"[INFO] Manual control active for Freeway RAM debugging.")
    print(f"[INFO] Controls: 'w' (UP), 's' (DOWN), 'q' (QUIT), 'p' (PAUSE/UNPAUSE), 'v' (SAVE RAM state).")
    print(f"[INFO] Goal: Confirm Chicken Y (RAM[{player_chicken_y_candidate_py_idx}] / 0x{STELLA_RAM_BASE_ADDR + player_chicken_y_candidate_py_idx:02x}), Score, and Car property blocks.")
    print(f"[INFO] **For CARS: Observe blocks of 4 bytes. Within each block, one byte is likely X-pos (constantly changing/wrapping), others are Y/Lane, Speed, or State.**")
    print(f"[INFO] Use 'p' to PAUSE and inspect RAM carefully, and 'v' to save console output.\n")

    frame_count = 0
    current_action = 0 # Start with NOOP
    paused = False

    log_file_name = "freeway_ram_log.csv"
    log_file = open(log_file_name, 'w', newline='')
    csv_writer = csv.writer(log_file)
    header = ['frame', 'action_taken', 'reward', 'terminated', 'truncated'] + [f'ram_{i}' for i in range(128)]
    csv_writer.writerow(header)

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print(f"[INFO] Game {'PAUSED' if paused else 'UNPAUSED'}")
                if paused:
                    cv2.waitKey(0)
            elif key == ord("v"):
                print(f"\n=== SAVED RAM STATE (Frame {frame_count}) ===")
                for i in range(0, 128, 8):
                    row = " ".join(f"{ram[j]:3d}" for j in range(i, min(i+8, 128)))
                    print(f"RAM[{i:2d}-{min(i+7, 127):2d}]: {row}")
                print()

            action_changed_this_frame = False
            if key == ord('w'): current_action = 1
            elif key == ord('s'): current_action = 2
            else:
                if current_action in [1, 2] and key == -1:
                    current_action = 0

            if paused:
                overlay = obs.copy()
                y_offset = 20
                cv2.putText(overlay, f"Score: {info.get('score', 'N/A')}", (10, y_offset), font, 1.0, (255, 255, 0), 1)
                cv2.putText(overlay, f"Action: {current_action}", (10, y_offset + 36), font, 1.0, (255, 255, 255), 1)
                cv2.putText(overlay, "PAUSED", (obs.shape[1] // 2 - 30, 10), font, 1.0, (0, 0, 255), 2)

                # Display Chicken Y
                if 0 <= player_chicken_y_candidate_py_idx < 128:
                    cv2.putText(overlay, f"P_Y: {ram[player_chicken_y_candidate_py_idx]}", (10, y_offset + 54), font, 1.0, (0, 255, 255), 1)

                # Display Car Blocks on overlay
                car_overlay_y = y_offset + 72
                for block_name, indices in car_blocks_display.items():
                    block_values = []
                    for py_idx in indices:
                        if 0 <= py_idx < 128:
                            block_values.append(f"0x{ram[py_idx]:02x}") # Display in hex for compact view
                        else:
                            block_values.append("ERR")
                    cv2.putText(overlay, f"{block_name}: [{' '.join(block_values)}]", (10, car_overlay_y), font, 1.0, (255, 165, 0), 1)
                    car_overlay_y += 18

                resized = cv2.resize(overlay, (obs.shape[1] * 3, obs.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Freeway RAM Mapper", resized)
                continue


            obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            ram = env.unwrapped.ale.getRAM()

            frame_count += 1

            # === Detect ALL changes ===
            changed = [(i, int(prev_ram[i]), int(ram[i])) for i in range(128) if ram[i] != prev_ram[i]]
            increasing = [(i, old, new) for i, old, new in changed if new > old]
            decreasing = [(i, old, new) for i, old, new in changed if new < old]

            # === Log current state to CSV ===
            row_data = [frame_count, current_action, reward, terminated, truncated] + list(ram)
            csv_writer.writerow(row_data)

            # === Display current key values (console) ===
            if frame_count % 30 == 0 or changed:
                print(f"\n=== Frame {frame_count} (Action: {current_action}) ===")
                print(f"Score from info: {info.get('score', 'N/A')}")
                print(f"Lives from info: {info.get('lives', 'N/A')}")

                print(f"RAM[{general_timer_py_idx}] (Frame Timer?): {ram[general_timer_py_idx]} (Stella 0x{STELLA_RAM_BASE_ADDR + general_timer_py_idx:02x})")
                print(f"Score? RAM[{score_candidate_1_py_idx}]: {ram[score_candidate_1_py_idx]} (0x{STELLA_RAM_BASE_ADDR + score_candidate_1_py_idx:02x}) | RAM[{score_candidate_2_py_idx}]: {ram[score_candidate_2_py_idx]} (0x{STELLA_RAM_BASE_ADDR + score_candidate_2_py_idx:02x})")

                # --- Player Chicken Y (Confirmed if you verify it) ---
                if 0 <= player_chicken_y_candidate_py_idx < 128:
                    print(f"P_Y (Your Candidate): RAM[{player_chicken_y_candidate_py_idx}] (0x{STELLA_RAM_BASE_ADDR + player_chicken_y_candidate_py_idx:02x}): {ram[player_chicken_y_candidate_py_idx]}")

                # --- Car Blocks in Console ---
                print("\n--- Car Property Blocks (Look for consistent, rapid changes) ---")
                for block_name, indices in car_blocks_display.items():
                    block_values_str = []
                    for py_idx in indices:
                        if 0 <= py_idx < 128:
                            block_values_str.append(f"RAM[{py_idx}] (0x{STELLA_RAM_BASE_ADDR + py_idx:02x}): 0x{ram[py_idx]:02x}")
                        else:
                            block_values_str.append(f"RAM[INVALID_IDX] (0x{STELLA_RAM_BASE_ADDR + py_idx:02x}): Out of Bounds!")
                    print(f"{block_name}: " + " | ".join(block_values_str))

                print("\n--- ALL CHANGES (Look for other car properties here) ---")
                if increasing:
                    print(f"[INCREASING] {increasing[:10]}")
                if decreasing:
                    print(f"[DECREASING] {decreasing[:10]}")
                print("-------------------------------------------------------")

            # === Look for specific patterns ===
            if reward > 0:
                print(f"[REWARD] Got reward {reward}! Check RAM changes: {[(i, int(prev_ram[i]), int(ram[i])) for i in range(128) if ram[i] != prev_ram[i]][:5]}")

            prev_ram = ram.copy()

            # === Create visual overlay ===
            overlay = obs.copy()
            y_offset = 20

            cv2.putText(overlay, f"Score: {info.get('score', 'N/A')}", (10, y_offset), font, 1.0, (255, 255, 0), 1)
            cv2.putText(overlay, f"Lives: {info.get('lives', 'N/A')}", (10, y_offset + 18), font, 1.0, (255, 255, 0), 1)
            cv2.putText(overlay, f"Action: {current_action}", (10, y_offset + 36), font, 1.0, (255, 255, 255), 1)

            # --- Chicken Y on overlay ---
            if 0 <= player_chicken_y_candidate_py_idx < 128:
                cv2.putText(overlay, f"P_Y: {ram[player_chicken_y_candidate_py_idx]}", (10, y_offset + 54), font, 1.0, (0, 255, 255), 1)
            else:
                cv2.putText(overlay, f"P_Y: Index Error!", (10, y_offset + 54), font, 1.0, (0, 0, 255), 1)

            # --- Car Blocks on overlay ---
            car_overlay_y = y_offset + 72
            for block_name, indices in car_blocks_display.items():
                block_values = []
                for py_idx in indices:
                    if 0 <= py_idx < 128:
                        block_values.append(f"0x{ram[py_idx]:02x}")
                    else:
                        block_values.append("ERR")
                cv2.putText(overlay, f"{block_name}: [{' '.join(block_values)}]", (10, car_overlay_y), font, 1.0, (255, 165, 0), 1)
                car_overlay_y += 18

            if paused:
                cv2.putText(overlay, "PAUSED", (obs.shape[1] // 2 - 30, 10), font, 1.0, (0, 0, 255), 2)


            for j, (i, old_val, new_val) in enumerate(changed[:10]):
                color = (0, 255, 0) if new_val > old_val else (0, 0, 255)
                txt = f"RAM[{i}] {old_val} -> {new_val}"
                cv2.putText(overlay, txt, (200, y_offset + j * 18), font, 1.0, color, 1)

            resized = cv2.resize(overlay, (obs.shape[1] * 3, obs.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Freeway RAM Mapper", resized)

            if done:
                print(f"[GAME OVER] Final score: {info.get('score', 'N/A')}")
                obs, info = env.reset()
                frame_count = 0
                current_action = 0


    finally:
        env.close()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"RAM log saved to {log_file_name}")

if __name__ == "__main__":
    main()