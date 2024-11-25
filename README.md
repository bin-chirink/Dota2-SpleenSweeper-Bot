# Spleen Sweeper AI Bot for Dota 2

🖱️ **Spleen Sweeper Bot** is an AI-powered script designed for the Dota 2 mini-game, Spleen Sweeper, which is similar to Minesweeper. The bot efficiently completes all 5 difficulty levels, providing a smooth and effortless gaming experience.

![Spleen Sweeper](https://github.com/bin-chirink/Dota2-SpleenSweeper-Bot/blob/main/resource/preview.gif?raw=true)

---

## Features

- **AI-Powered Gameplay**: The only bot for the *Minesweeper* mini-game in Dota 2 Crownfall Act 4, utilizing image recognition and advanced algorithms.
- **User-Friendly**: Minimal setup required to start automating the gameplay.
- **Smart Controls**: Automatically detects the game board and makes optimal moves.
- **Seamless Integration**: Works flawlessly with Dota 2 in borderless windowed mode.

---

## Download and Usage

1. **Download the Bot**: [Releases](https://github.com/bin-chirink/Dota2-SpleenSweeper-Bot/releases/)
2. **Extract the Archive**: Unzip the downloaded file (Password: `369411`).
3. **Prepare Your Game**: Set Dota 2 resolution to **1920x1080 (16:9)** in borderless windowed mode.
4. **Start the Bot**: Run `run_agent.exe` as administrator.
5. **Launch the Mini-Game**: Navigate to the Minesweeper game page in Dota 2.
6. **Control the Bot**:
   - Hover the mouse over **Play** on the game page and press `B` to start.
   - After completing a level, hover over **Continue** and press `B`.
   - Press `Q` to pause/unpause the bot.
   - Press `Esc` to exit the program.

---

## Requirements

- **Downloading / Installing Issues**:  
  If you encounter problems with downloading or installing the archive, follow these steps:
  1. **Disable or Remove Antivirus**: Some antivirus programs may flag the file (it is completely clean).
  2. **Try a Different Browser**: If you can’t download, copy the link and use another browser.
  3. **Disable Windows SmartScreen**: Also, ensure the **Visual C++ package** is up to date.

---

## Q&A

**Will it get you VAC banned?**  
The program uses image recognition and does not access any game files, so it should be safe to use.

**After pressing `B`, it performs no action or random clicks.**  
This happens if the board wasn’t recognized correctly. Try replacing `topLeft.jpg` and `botRight.jpg` in the resource folder with your own screenshots. Make sure the center points of both images correspond to the top-left and bottom-right corners of the board.

**Clicked on a mine or freezing when a level is not finished.**  
If the program works fine most of the time, it’s likely that a cell number wasn’t recognized accurately. Try pressing `T` to turn off turbo mode, or simply retry a few more times.

**Cleared all levels, but the score wasn’t enough.**  
Try again.
