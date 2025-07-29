import gymnasium as gym
import numpy as np
import cv2
import random
import csv
import ale_py

def main():
    env = gym.make("ALE/Seaquest-v5", render_mode="rgb_array", frameskip=1)
    obs, info = env.reset()
    prev_ram = env.unwrapped.ale.getRAM().copy()

    font = cv2.FONT_HERSHEY_PLAIN

    # --- Confirmed RAM Addresses ---
    player_x_confirmed_addr = 70
    player_y_confirmed_addr = 97 # Confirmed!

    oxygen_ram_addr = 102
    frame_timer_ram_addr = 1

    diver_sync_group_addrs = [31, 32, 33]

    # --- New Candidate for "Diver Rescued" ---
    diver_rescued_count_candidate = 62 # <--- This is where RAM[62] is defined

    print(f"[INFO] Manual control active for finding 'diver rescued' flag.")
    print(f"[INFO] Player X: RAM[{player_x_confirmed_addr}] | Player Y: RAM[{player_y_confirmed_addr}] | Oxygen: RAM[{oxygen_ram_addr}]")
    print(f"[INFO] **WATCH 'Diver Count? RAM[{diver_rescued_count_candidate}]' IN THE OVERLAY AND CONSOLE.**") # <--- Message for console
    print("[INFO] Expect it to change (e.g., increment) when a diver is picked up.")
    print("[INFO] Controls: 'w' (UP), 'a' (LEFT), 's' (DOWN), 'd' (RIGHT), 'space' (FIRE).")
    print("[INFO] Other: 'q' (QUIT), 'p' (PAUSE/UNPAUSE), 'v' (SAVE current RAM state to console).\n")

    frame_count = 0
    current_action = 0 # Start with NOOP
    paused = False # New state for pause functionality

    log_file_name = "seaquest_ram_diver_log.csv"
    log_file = open(log_file_name, 'w', newline='')
    csv_writer = csv.writer(log_file)
    header = ['frame', 'action_taken', 'reward', 'terminated', 'truncated'] + [f'ram_{i}' for i in range(128)]
    csv_writer.writerow(header)

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
        if key == ord('w'):
            current_action = 2 # UP
            action_changed_this_frame = True
        elif key == ord('a'):
            current_action = 4 # LEFT
            action_changed_this_frame = True
        elif key == ord('s'):
            current_action = 5 # DOWN
            action_changed_this_frame = True
        elif key == ord('d'):
            current_action = 3 # RIGHT
            action_changed_this_frame = True
        elif key == ord(' '):
            current_action = 1 # FIRE
            action_changed_this_frame = True

        if current_action == 1 and not action_changed_this_frame and key == -1:
            current_action = 0


        if paused:
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

        # === Display current key values (every 15 frames or when changes occur) ===
        if frame_count % 15 == 0 or changed:
            print(f"\n=== Frame {frame_count} (Action: {current_action}) ===")
            print(f"Score from info: {info.get('score', 'N/A')}")
            print(f"Lives: {info.get('lives', 'N/A')}")

            print(f"RAM[{oxygen_ram_addr}] (oxygen): {ram[oxygen_ram_addr]}")
            print(f"RAM[{frame_timer_ram_addr}] (frame timer): {ram[frame_timer_ram_addr]}")
            print(f"Player X: RAM[{player_x_confirmed_addr}]: {ram[player_x_confirmed_addr]}")
            print(f"Player Y: RAM[{player_y_confirmed_addr}]: {ram[player_y_confirmed_addr]}")

            # --- Display new candidate for Diver Rescued Count in console ---
            print(f"Diver Count? RAM[{diver_rescued_count_candidate}]: {ram[diver_rescued_count_candidate]}")

            print("Diver sync group:", " | ".join([f"RAM[{addr}]: {ram[addr]}" for addr in diver_sync_group_addrs]))

            if increasing:
                print(f"[INCREASING] {increasing[:10]}")
            if decreasing:
                print(f"[DECREASING] {decreasing[:10]}")

        if reward > 0:
            print(f"[REWARD] Got reward {reward}! Check these changes: {changed}")

        if ram[oxygen_ram_addr] < prev_ram[oxygen_ram_addr]:
            print(f"[OXYGEN?] RAM[{oxygen_ram_addr}] decreased: {prev_ram[oxygen_ram_addr]} → {ram[oxygen_ram_addr]}")

        prev_ram = ram.copy()

        # === Create visual overlay ===
        overlay = obs.copy()
        y_offset = 20

        cv2.putText(overlay, f"Oxygen: {ram[oxygen_ram_addr]}", (10, y_offset), font, 1.0, (255, 255, 0), 1)
        cv2.putText(overlay, f"Score: {info.get('score', 'N/A')}", (10, y_offset + 18), font, 1.0, (255, 255, 0), 1)
        cv2.putText(overlay, f"Lives: {info.get('lives', 'N/A')}", (10, y_offset + 36), font, 1.0, (255, 255, 0), 1)
        cv2.putText(overlay, f"P_X: {ram[player_x_confirmed_addr]}", (10, y_offset + 54), font, 1.0, (0, 255, 255), 1)
        cv2.putText(overlay, f"P_Y: {ram[player_y_confirmed_addr]}", (10, y_offset + 72), font, 1.0, (0, 255, 255), 1)
        cv2.putText(overlay, f"Action: {current_action}", (10, y_offset + 90), font, 1.0, (255, 0, 255), 1)

        # --- Display Diver Rescued Count on Overlay ---
        cv2.putText(overlay, f"Divers: {ram[diver_rescued_count_candidate]}", (10, y_offset + 108), font, 1.0, (0, 255, 0), 1) # <--- This line adds it to the overlay


        for j, (i, old_val, new_val) in enumerate(changed[:10]):
            color = (0, 255, 0) if new_val > old_val else (0, 0, 255)
            txt = f"RAM[{i}] {old_val} -> {new_val}"
            cv2.putText(overlay, txt, (200, y_offset + j * 18), font, 1.0, color, 1)

        resized = cv2.resize(overlay, (obs.shape[1] * 3, obs.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Seaquest RAM Mapper", resized)

        if done:
            print(f"[GAME OVER] Final score: {info.get('score', 'N/A')}")
            obs, info = env.reset()
            frame_count = 0
            current_action = 0


    env.close()
    cv2.destroyAllWindows()
    log_file.close()
    print(f"RAM log saved to {log_file_name}")

if __name__ == "__main__":
    main()