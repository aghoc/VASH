' VASH app launcher for Windows: runs app\server.py with a hidden console (no terminal window).
' The hidden console keeps the safe stop (CTRL_BREAK) to the hand and EMG programs working.
' If the app is already running, server.py only opens the browser again. Quit from the [앱 끄기] button.
Set fs = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
emg = fs.GetParentFolderName(fs.GetParentFolderName(WScript.ScriptFullName))
sh.CurrentDirectory = emg
sh.Run """" & emg & "\.venv\Scripts\python.exe"" app\server.py", 0, False
