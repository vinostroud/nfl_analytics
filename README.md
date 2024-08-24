## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

### About the Project
This project is a simple NFL analytics dashboard built with Streamlit. It allows users to request visualizion of some NFL data key through interactive charts. Data is pulled from https://github.com/nflverse/nfl_data_py. 

### Features
As of August 2024, the charts show:

-Team EPA for first down
-Team EPA for first and second down
-Comparison of EPA, points scored, and turned over
-A simple regression (though as of this writing, this needs to be improved)

### Installation
To get a local copy up and running, follow these steps:

### Prerequisites
- Python 3.x (3.12 is recommended)
- Poetry (for package management)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/vinostroud/nfl_analytics
   ```
2. Navigate to the project directory:
   ```bash
   cd nfl-analytics
   ```
3. Install the dependencies using Poetry:
   ```bash
   poetry install
   ```
4. Run the Streamlit app:
   ```bash
   poetry run streamlit run src/app_fe.py
   ```

### Usage
After installing the project, you can run the Streamlit app with the following command:
```bash
poetry run streamlit run src/app_fe.py
```
After running the app in Streamlit, follow instructions in your terminal window to view the app in your browser. Typically you will navigate to `http://localhost:8501`


### **Folder Structure**
-`src/` contains the main app code
-`tests/` contains test scripts using pytest

### **Technologies Used**
- Python
- Streamlit
- Pandas
- NFL Data Py (nfl_data_py)
- Sklearn

### Contributing
This project was created for fun and out of an interest in NFL data. Contributions, PRs, creation of issues, or ideas are therefore welcome.

### License



