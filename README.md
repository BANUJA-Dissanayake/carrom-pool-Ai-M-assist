# Carrom Pool AI Assistant 🎯

A real-time, high-FPS visual aim assistant for **Carrom Pool** (Miniclip) running on PC Android emulators (BlueStacks, LDPlayer, Nox, etc.).

Powered entirely by **Computer Vision (OpenCV)** and **2D Physics Raycasting** via screen capture. It does **not** hook into game memory or modify game files.

---

## ⚡ Features

- **High Performance (100–200+ FPS):** Ultra-fast frame processing (~1.6–3.5 ms per frame) for smooth, zero-latency real-time aim tracking.
- **Accurate Trajectory:** Uses collinear PCA / `fitLine` on the aim arrow to match the game's actual trajectory within 0.1°.
- **Clean Idle State:** Zero assist lines when resting or sliding the striker along the baseline. Lines only appear when you actively drag the aim arrow.
- **Target Black Coins Only:** Specifically tuned to target black coins, automatically ignoring white coins and wooden board circles.
- **Dual-Layer Neon Laser Lines:**
  - 🟡 **Neon Yellow/Gold Line:** Primary striker aim path.
  - 🟢 **Neon Green Line:** Target coin trajectory when aimed directly or via bank shot into a pocket.
  - 🔵 **Neon Cyan Line:** Target coin trajectory when not pocketed.
  - 🟠 **Neon Orange Line:** 90° striker deflection trajectory after collision.
- **Physics & Cushion Rebounds:** Realistic cushion bounce reflections accounting for ball radius, ghost ball contact circles, and pocket detection.
- **Custom Interactive Selection:** Drag and select your emulator's board area once at startup.

---

## 📋 Requirements

- **Operating System:** Windows 10 / 11
- **Python:** 3.8 or higher
- **Android Emulator:** BlueStacks, LDPlayer, Nox, etc. running Carrom Pool

---

## 🚀 Installation

1. **Clone or Download the Repository:**
   ```bash
   git clone https://github.com/BANUJA-Dissanayake/carrom-pool-ai.git
   cd carrom-pool-ai
   ```

2. **Install Required Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Or using the Python launcher on Windows: `py -m pip install -r requirements.txt`)*

---

## 🎮 How to Run

1. **Start Carrom Pool** in your emulator and enter a match or practice mode.
2. **Run the assistant:**
   ```bash
   python main.py
   ```
   *(Or on Windows: `py main.py`)*
3. **Select Board Area:**
   - A screenshot window will pop up.
   - **Click and drag a box** around only the square Carrom board.
   - Press <kbd>Enter</kbd> or <kbd>Space</kbd> to confirm.
4. **Play:**
   - The assistant window will open showing your board.
   - Drag the red aim arrow in the game to see real-time laser trajectory lines, ghost balls, and pocket indicators.
5. **Exit:**
   - Press <kbd>Q</kbd> in the assistant window anytime to exit.

---

## 📁 Project Structure

```text
carrom-pool-ai/
│
├── main.py              # Application entry point & display loop
├── screen_capture.py    # High-speed screen grabber (MSS) & interactive ROI selection
├── board_detector.py    # Striker, aim arrow, and black coin detection
├── shot_calculator.py   # Raycasting physics, cushion bounces, and laser rendering
├── calibrate.py         # Color tuning helper tool
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚙️ How It Works

1. **Screen Capture:** Uses the `mss` library to grab the selected board region at monitor refresh rate.
2. **Striker Tracking:** Detects the striker disc on the baseline using Hough transform and brightness scoring.
3. **Arrow & Aim Vector:** Isolates the red/orange aim arrow using color thresholding, eliminates resting glow rings, and computes the exact trajectory angle using `cv2.fitLine`.
4. **Raycasting Physics:** Traces the striker's forward path, calculates cushion reflections, determines contact points with black coins, computes the ghost ball position, and predicts whether the coin will sink into a corner pocket.

---

## ⚠️ Disclaimer

This project is an open-source educational computer vision experiment demonstrating real-time visual tracking and geometry-based raycasting. It operates strictly via external screen capture and does not alter, inject, or manipulate game code or memory.


from BANUJA-Dissanayake