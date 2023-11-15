# Web-based GUI for Machine Learning Workflow/Pipeline

Professional software development project from the Fall 2023 semester at UAF.

## Development Instructions

#### Flask Application

###### Local Debugging

> [!NOTE]
> This project utilizes [`Pipenv`](https://pipenv.pypa.io/en/latest/) for managing virtual environments.
> You can install it with `pip install --user pipenv`.

- Clone repository: `gh repo clone jbonda/ml-workflow-pipeline`
- Navigate to the source directory: `cd ml-workflow-pipeline/ML_GUI`
- Install [dependencies](ML_GUI/Pipfile): `pipenv shell`
- Run program: `flask --app server.py --debug run`

###### Production Environment

- Prerequisites: [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) → Linux
- Create virtual environment: `conda create --name {name} python={version}`
- Activate virtual environment: `conda activate {name}`
- Clone repository: `git clone https://github.com/jbonda/ml-workflow-pipeline.git`
- Navigate to the source directory: `cd ml-workflow-pipeline/ML_GUI`
- Install [dependencies](ML_GUI/Pipfile): `conda install -c anaconda {package}`
- Run program: `gunicorn --config gunicorn_config.py app:app`

<details>
<summary>System Benchmarking</summary>

###### Node.js

- Navigate to the source directory: `cd benchmark/src`
- Install dependencies: `npm i`
- Run the development script: `npm run devstart`

###### Hybrid

- Supplementary resources are available within the [`benchmark`](https://github.com/jbonda/ml-workflow-pipeline/tree/main/benchmark) directory.

###### .NET

- [ML.NET Tutorial - Get started in 10 minutes](https://dotnet.microsoft.com/en-us/learn/ml-dotnet/get-started-tutorial/intro)

## Project Diagrams

![PERT/CPM Chart](docs/PERT_CPM_Chart.svg)

</details>
