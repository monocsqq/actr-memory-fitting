# README

This project assumes a **Windows x86_64 environment**.

------------------------------------------------------------------------

# 1. Setup

## 1.1 ACT-R Environment

### Install SBCL

``` bash
winget install SBCL.SBCL
```

### Install QuickLisp

``` bash
sbcl --load quicklisp.lisp
```

### Recompile ACT-R

``` bash
sbcl --eval "(defparameter *given-port* 12345)" --load .\actr7.x\recompile-act-r.lisp
```

-   The port number can be any value.
-   It is used for parallel connection tests.

### Verify ACT-R Installation

``` bash
sbcl --eval "(defparameter *given-port* 12345)" --load .\actr7.x\recompile-act-r.lisp
```

If the following message appears:

    ######### Loading of ACT-R 7 is complete #########

the installation was successful.

------------------------------------------------------------------------

## 1.2 Python Environment

The following versions were used in this study:

    python: 3.11.4
    numpy: 1.24.3
    pandas: 1.5.3
    Levenshtein: 0.26.1
    cv2: 4.10.0

------------------------------------------------------------------------

# 2. Model Simulation

## 2.1 Launch ACT-R Instances

Run the appropriate batch file depending on the experiment:

-   **Experiment 1 simulation**

        run_100_actr(expt1).bat

-   **Experiment 2 simulation**

        run_100_actr_ori_grouped(expt2).bat

------------------------------------------------------------------------

## 2.2 Run Parameter Search

Execute `search_param.py` in the `python` directory.

### Select Input Dataset

Edit `search_param.py` and uncomment the relevant line:

-   **Experiment 1**

    ``` python
    target = '20240303_listonly_normalized'
    ```

-   **Experiment 2**

    -   Analysis 1:

        ``` python
        target = '20241225_screened'
        ```

    -   Analysis 2:

        -   Arousal:

            ``` python
            target = '20241225_divaro_screened'
            ```

        -   Valence:

            ``` python
            target = '20241225_divemo_screened'
            ```

------------------------------------------------------------------------

### Execute

``` bash
python .\python\search_param.py (trial_name)
```

`trial_name` specifies the name of the parameter estimation run.

## Related publication

This repository contains code associated with the following conference paper:

K. Shimbori et al. (2025). Emotional Parameters in Cognitive Architecture: Examination Through Simple Memory Performance. Proceedings of the Annual Meeting of the Cognitive Science Society.
[https://escholarship.org/uc/item/17c1t2d1]
