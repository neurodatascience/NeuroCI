# NeuroCI
NeuroCI is a lightweight Python framework for automating neuroimaging pipeline execution on HPC systems using Nipoppy, with integrated CI/CD capabilities for result processing and tracking.
Its purpose is to facilitate carrying out neuroimaging experiments across multiple pipelines and datasets simultaneously, in order to assess result robustness and replicability across analytical variations.

Note that this branch is being used to develop NeuroCI v2.0.
v2.0 is marked by a number of changes from the original architecture, relying on direct communication with Slurm/SGE HPCs and using Nipoppy rather than leveraging the CBRAIN Web Computation Platform. The old, functional (as of March 2025) CBRAIN version of NeuroCI (v1.0) as described in the publication [(Sanz-Robinson, 2022)](https://ieeexplore.ieee.org/document/9973641) can be found in the 'old-version-cbrain' branch.

## Usage Instructions

**Note**: Prior to use, please be aware that this experiment repository shares multiple files from your Nipoppy datasets (in the 'experiment_state' directory), including the manifest and bagel files containing subject IDs, and the global configuration file with potential sensitive path information. Ensure you review and manage sensitive data accordingly.

### Prerequisites
0. **Dataset Setup**: Initialize your BIDS-format dataset(s) on the HPC with all required pipelines using [Nipoppy](https://nipoppy.readthedocs.io/en/latest/).

### Setup Steps
1. **Repository Setup**: Fork the 'blank template' branch (for customization) or refer to 'master' for a working example.
2. **SSH Configuration**:
   - Generate an SSH key pair (`ssh-keygen`)
   - Install the public key on your HPC
3. **Connection Setup**:
   - Modify `config_files/ssh_config` with your HPC connection details (supports proxyjump/proxycommand)
4. **GitHub Secrets**:
   - Add repository secrets:
     - `SSH_CONFIG_PATH`: Path to your SSH config file
     - `SSH_PRIVATE_KEY`: Your private SSH key
5. **User Scripts**:
   - Place lightweight Python scripts in `user_scripts/` directory
   - Scripts should:
     - Process IDP outputs from `/tmp/neuroci_idp_state/` (directory for downloaded Nipoppy IDP outputs)
     - Save results to `experiment_root/experiment_state/` (recommended, but can be saved elsewhere)
     - Declare dependencies in `requirements.txt`
6. **Experiment Configuration**:
   - Edit `experiment_definition.yaml` to specify:
     - Nipoppy Dataset paths
     - Pipeline names/versions
     - User script filenames (to be executed in the order provided).
     - Target host and scheduler (slurm/sge)
   - Modifying this file triggers the Github Actions CI run.
8. **Scheduling (Optional)**:
   - Modify `.github/workflows/ci_trigger.yml` for periodic execution

## Code Overview

### Core Components

#### `experiment.py`
- **Experiment**: Main class orchestrating the workflow
  - Validates experiment definition
  - Manages dataset compliance checks
  - Coordinates pipeline execution and result extraction
  - Handles SSH connections and file operations

#### `ssh_utils.py`
- **SSHConnectionManager**: Secure remote connection handler
  - Manages SSH connections with proxy support
  - Executes remote commands (including Nipoppy commands)
  - Performs dataset compliance verification

#### `file_utils.py`
- **FileOperations**: Local file management
  - Syncs experiment state from HPC to local repo
  - Executes user processing scripts
  - Handles Git operations for result tracking

### Supporting Files

#### `main.py`
- Entry point that:
  - Loads experiment definition
  - Initializes and executes the Experiment
  - Coordinates the complete workflow

## Known Limitations (Stuff to address)
- I will create a user_template branch for users to fork without all of my personal configs cluttering it up.
- We are limited to the file invocations for state backup being called 'invocation.json' as a generic step name. This will (eventually) be addressed by importing some Nipoppy code.
- Some users will not want their paths and private stuff to be exposed in the experiment definition and SSH config files. This will (eventually) be addressed by creating the option to provide all this information as a CI secret instead of reading from a file.
