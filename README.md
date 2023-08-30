# Domain Adaptation for Remote Sensing in Scene Classification: An Investigation into Adaptive Dropout and Adaptive Batch Normalization Techniques

This repository contains code for domain adaptation experiments with the focus on adapting between AID (Aerial Image Dataset) and UCM (UC Merced Land Use Dataset).

## 📄 Documentation

The project's full report, containing the research methods, datasets used, and results can be found in [`AdaptiveDA_report.pdf`](AdaptiveDA_report.pdf).

## 📂 Directory Structure

- `src/dann.py`: Domain-Adversarial Neural Network implementation
- `src/dataset_creation.py`: Script for dataset pre-processing and data loaders
- `src/deepcoral.py`: Implementation of Deep CORAL (CORrelation ALignment)
- `src/lower_bound.py`: Lower bound
- `src/upper_bound.py`: Upper bound
- `src/utils.py`: Utility functions including model architectures and helper functions

## 📦 Requirements

You can install the required packages using the following command:

\`\`\`
pip install -r requirements.txt
\`\`\`

## 🎯 Datasets

- [AID: Aerial Image Dataset](https://captain-whu.github.io/AID/)
- [UCM: UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## 🛠️ Usage

1. Clone the repository:

\`\`\`
git clone https://github.com/yourusername/YourRepositoryName.git
\`\`\`

2. Install the requirements:

\`\`\`
pip install -r requirements.txt
\`\`\`

3. Create a folder named 'data' and install the required datasets there.

\`\`\`
mkdir data
\`\`\`

3. Run the script for the adaptation method you're interested in. For example, to run DANN:

\`\`\`
cd src
python dann.py
\`\`\`

4. The output results will be displayed on the console and can be further analyzed as per your requirements.

## 📈 Results

### AID to UCM Transfer

| **AD** | **AdaBN** | **DANN**  | **DeepCORAL** |
|--------|-----------|-----------|----------------|
| 0      | 0         | 64.92%    | 64.45%         |
| 0      | 1         | 63.88%    | 66.87%         |
| 1      | 0         | 71.52%    | 69.53%         |
| 1      | 1         | 68.66%    | 62.23%         |

*Table 1: Results of our models for AID to UCM transfer with various configurations of Adaptive Dropout (AD) and Adaptive Batch Normalization (AdaBN). 1: True, 0: False.*

### UCM to AID Transfer

| **AD** | **AdaBN** | **DANN**  | **DeepCORAL** |
|--------|-----------|-----------|----------------|
| 0      | 0         | 60.64%    | 61.03%         |
| 0      | 1         | 55.71%    | 56.86%         |
| 1      | 0         | 64.61%    | 62.37%         |
| 1      | 1         | 49.85%    | 65.11%         |

*Table 2: Results of our models for UCM to AID transfer with various configurations of Adaptive Dropout (AD) and Adaptive Batch Normalization (AdaBN). 1: True, 0: False.*

For detailed methodology and discussion, please refer to [`AdaptiveDA_report.pdf`](AdaptiveDA_report.pdf).

## 📧 Contact

If you have any questions or would like to contribute, feel free to reach out.
- Emirhan Böge: [emirhanboge@sabanciuniv.edu](mailto:emirhanboge@sabanciuniv.edu)
