@echo off
REM ============================================================================
REM Handwritten Number Recognition System (HNRS) Launcher
REM ============================================================================
REM This batch file provides easy access to run the GUI and backend components
REM of the Handwritten Number Recognition System on Windows.
REM ============================================================================

setlocal enabledelayedexpansion

REM Set colors for output
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

REM Display header
echo %BLUE%============================================================================%RESET%
echo %BLUE%    Handwritten Number Recognition System (HNRS) Launcher%RESET%
echo %BLUE%============================================================================%RESET%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python is not installed or not in PATH%RESET%
    echo Please install Python 3.7+ and add it to your system PATH
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Display Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%Python %PYTHON_VERSION% detected%RESET%
echo.

REM Check if we're in the correct directory
if not exist "src\main.py" (
    echo %RED%ERROR: src\main.py not found%RESET%
    echo Please run this script from the project root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

if not exist "src\gui.py" (
    echo %RED%ERROR: src\gui.py not found%RESET%
    echo Please ensure all project files are present
    pause
    exit /b 1
)

REM Main menu
:MENU
echo %YELLOW%Please select an option:%RESET%
echo.
echo %BLUE%1.%RESET% Launch GUI Application
echo %BLUE%2.%RESET% Run Backend Tests
echo %BLUE%3.%RESET% Check Dependencies
echo %BLUE%4.%RESET% Test Single Image (CLI)
echo %BLUE%5.%RESET% Compare All Models (CLI)
echo %BLUE%6.%RESET% Launch GUI + Show Backend Info
echo %BLUE%7.%RESET% Help / Usage Information
echo %BLUE%8.%RESET% Exit
echo.
set /p choice=%YELLOW%Enter your choice (1-8): %RESET%

if "%choice%"=="1" goto GUI_ONLY
if "%choice%"=="2" goto BACKEND_TESTS
if "%choice%"=="3" goto CHECK_DEPS
if "%choice%"=="4" goto TEST_IMAGE
if "%choice%"=="5" goto COMPARE_MODELS
if "%choice%"=="6" goto GUI_WITH_INFO
if "%choice%"=="7" goto HELP
if "%choice%"=="8" goto EXIT

echo %RED%Invalid choice. Please enter a number between 1-8.%RESET%
echo.
goto MENU

REM ============================================================================
REM Option 1: Launch GUI Application Only
REM ============================================================================
:GUI_ONLY
echo.
echo %GREEN%Launching GUI Application...%RESET%
echo %YELLOW%This will open the Handwritten Number Recognition GUI%RESET%
echo.
python src\main.py --gui
if errorlevel 1 (
    echo.
    echo %RED%GUI failed to start. Checking dependencies...%RESET%
    python src\main.py --check-deps
    echo.
    echo %YELLOW%Press any key to return to menu...%RESET%
    pause >nul
)
goto MENU

REM ============================================================================
REM Option 2: Run Backend Tests
REM ============================================================================
:BACKEND_TESTS
echo.
echo %GREEN%Running Backend Model Tests...%RESET%
echo %YELLOW%This will test all trained models and their functionality%RESET%
echo.
python src\main.py --test-models
echo.
echo %YELLOW%Press any key to return to menu...%RESET%
pause >nul
goto MENU

REM ============================================================================
REM Option 3: Check Dependencies
REM ============================================================================
:CHECK_DEPS
echo.
echo %GREEN%Checking System Dependencies...%RESET%
echo.
python src\main.py --check-deps
echo.
echo %YELLOW%Press any key to return to menu...%RESET%
pause >nul
goto MENU

REM ============================================================================
REM Option 4: Test Single Image
REM ============================================================================
:TEST_IMAGE
echo.
echo %GREEN%Test Single Image Recognition%RESET%
echo.
set /p image_path=%YELLOW%Enter path to image file: %RESET%

if not exist "%image_path%" (
    echo %RED%ERROR: Image file not found: %image_path%%RESET%
    echo.
    echo %YELLOW%Press any key to return to menu...%RESET%
    pause >nul
    goto MENU
)

echo.
echo %YELLOW%Select model for recognition:%RESET%
echo 1. CNN (Highest Accuracy - 99.23%%)
echo 2. SVM (Balanced Performance - 96.34%%)
echo 3. Random Forest (Fastest - 95.11%%)
echo.
set /p model_choice=%YELLOW%Enter choice (1-3): %RESET%

if "%model_choice%"=="1" set MODEL_NAME=cnn
if "%model_choice%"=="2" set MODEL_NAME=svm
if "%model_choice%"=="3" set MODEL_NAME=rf

if not defined MODEL_NAME (
    echo %RED%Invalid model choice%RESET%
    goto TEST_IMAGE
)

echo.
echo %GREEN%Processing image with %MODEL_NAME% model...%RESET%
python src\main.py --test-image "%image_path%" --model %MODEL_NAME%
echo.
echo %YELLOW%Press any key to return to menu...%RESET%
pause >nul
goto MENU

REM ============================================================================
REM Option 5: Compare All Models
REM ============================================================================
:COMPARE_MODELS
echo.
echo %GREEN%Model Comparison Mode%RESET%
echo.
set /p image_path=%YELLOW%Enter path to image file for comparison: %RESET%

if not exist "%image_path%" (
    echo %RED%ERROR: Image file not found: %image_path%%RESET%
    echo.
    echo %YELLOW%Press any key to return to menu...%RESET%
    pause >nul
    goto MENU
)

echo.
echo %GREEN%Testing image with all models...%RESET%
echo.
echo %BLUE%CNN Model (99.23%% accuracy):%RESET%
python src\main.py --test-image "%image_path%" --model cnn
echo.
echo %BLUE%SVM Model (96.34%% accuracy):%RESET%
python src\main.py --test-image "%image_path%" --model svm
echo.
echo %BLUE%Random Forest Model (95.11%% accuracy):%RESET%
python src\main.py --test-image "%image_path%" --model rf
echo.
echo %YELLOW%Press any key to return to menu...%RESET%
pause >nul
goto MENU

REM ============================================================================
REM Option 6: Launch GUI with Backend Information
REM ============================================================================
:GUI_WITH_INFO
echo.
echo %GREEN%Launching GUI with Backend Information...%RESET%
echo.
echo %BLUE%System Information:%RESET%
echo %YELLOW%- Project: Handwritten Number Recognition System%RESET%
echo %YELLOW%- Models Available: CNN, SVM, Random Forest%RESET%
echo %YELLOW%- Best Model: CNN (99.23%% accuracy)%RESET%
echo %YELLOW%- GUI Framework: tkinter%RESET%
echo %YELLOW%- Backend: Python CLI with multiple options%RESET%
echo.
echo %GREEN%Starting dependency check...%RESET%
python src\main.py --check-deps
echo.
echo %GREEN%Starting GUI application...%RESET%
python src\main.py --gui
if errorlevel 1 (
    echo.
    echo %RED%GUI failed to start. Please check the dependency report above.%RESET%
    echo.
    echo %YELLOW%Press any key to return to menu...%RESET%
    pause >nul
)
goto MENU

REM ============================================================================
REM Option 7: Help and Usage Information
REM ============================================================================
:HELP
echo.
echo %BLUE%============================================================================%RESET%
echo %BLUE%                    HNRS Help and Usage Information%RESET%
echo %BLUE%============================================================================%RESET%
echo.
echo %GREEN%SYSTEM OVERVIEW:%RESET%
echo The Handwritten Number Recognition System (HNRS) consists of:
echo.
echo %YELLOW%1. GUI Application (Frontend):%RESET%
echo    - Interactive drawing canvas for digit input
echo    - Image file upload functionality
echo    - Real-time number recognition
echo    - Model comparison interface
echo    - Preprocessing visualization
echo.
echo %YELLOW%2. Backend/CLI (Command Line Interface):%RESET%
echo    - Model testing and validation
echo    - Batch image processing
echo    - Dependency checking
echo    - Performance benchmarking
echo.
echo %GREEN%TRAINED MODELS:%RESET%
echo %YELLOW%- CNN Model:%RESET%        99.23%% accuracy (Best for complex patterns)
echo %YELLOW%- SVM Model:%RESET%        96.34%% accuracy (Balanced performance)
echo %YELLOW%- Random Forest:%RESET%    95.11%% accuracy (Fastest processing)
echo.
echo %GREEN%SUPPORTED IMAGE FORMATS:%RESET%
echo PNG, JPG, JPEG, BMP, TIFF, TIF
echo.
echo %GREEN%SYSTEM REQUIREMENTS:%RESET%
echo - Python 3.7+
echo - Required packages: opencv-python, tensorflow, scikit-learn, 
echo   pillow, matplotlib, numpy, pandas
echo.
echo %GREEN%MANUAL COMMANDS:%RESET%
echo %YELLOW%GUI Only:%RESET%           python src\main.py --gui
echo %YELLOW%Test Models:%RESET%        python src\main.py --test-models
echo %YELLOW%Check Dependencies:%RESET% python src\main.py --check-deps
echo %YELLOW%Test Image:%RESET%         python src\main.py --test-image ^<path^> --model ^<cnn^|svm^|rf^>
echo.
echo %GREEN%TROUBLESHOOTING:%RESET%
echo - If GUI fails to start, run "Check Dependencies" first
echo - Ensure all model files exist in src\models\ directory
echo - For image recognition issues, try different preprocessing options
echo - Check that image files are in supported formats
echo.
echo %YELLOW%Press any key to return to menu...%RESET%
pause >nul
goto MENU

REM ============================================================================
REM Exit
REM ============================================================================
:EXIT
echo.
echo %GREEN%Thank you for using the Handwritten Number Recognition System!%RESET%
echo %YELLOW%For more information, visit the project repository or documentation.%RESET%
echo.
pause
exit /b 0

REM ============================================================================
REM Error Handling
REM ============================================================================
:ERROR
echo.
echo %RED%An error occurred. Please check the following:%RESET%
echo - Python is properly installed and in PATH
echo - All project files are present
echo - Required dependencies are installed
echo.
echo %YELLOW%Run "Check Dependencies" option for detailed information.%RESET%
echo.
pause
goto MENU
