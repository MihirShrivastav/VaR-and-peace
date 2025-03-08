VaR and ES Forecasting using Structured Recurrent Neural Networks (SRNNs)

This repository contains the implementation of Structured Recurrent Neural Networks (SRNNs) for forecasting Value-at-Risk (VaR) and Expected Shortfall (ES), following the methodology outlined in the referenced research paper. The objective is to evaluate the predictive capacity of recurrent neural networks for financial risk forecasting, using SRNN-VE models trained on historical financial data.
1. Introduction: VaR and ES in Risk Management
Value-at-Risk (VaR)

VaR is a widely used measure in financial risk management that estimates the potential loss in the value of a portfolio over a given time horizon at a specified confidence level. Mathematically, for a confidence level αα:
P(L>VaRα)=1−α
P(L>VaRα​)=1−α

where LL is the portfolio loss.
Expected Shortfall (ES)

ES (also called Conditional VaR) provides an estimate of the expected loss beyond the VaR threshold. Unlike VaR, which only gives a cutoff point, ES accounts for the severity of losses in the tail:
ESα=E[L∣L>VaRα]
ESα​=E[L∣L>VaRα​]

It is considered a superior risk measure since it accounts for tail risk, which is particularly relevant for financial crises.
2. Research Basis: SRNNs for Risk Forecasting

The referenced research paper proposes Structured RNN models (SRNN-VE) to forecast dynamic VaR and ES estimates, using historical squared returns as input features. The paper presents three variations of SRNN models:

    SRNN-VE-1: Standard RNN model with a linear output layer.
    SRNN-VE-2: Incorporates a nonlinear transformation using square-root of hidden states.
    SRNN-VE-3: Hybrid approach combining the direct hidden state and its transformation.

Among these, SRNN-VE-3 was found to be the most effective, as it allows for nonlinear interactions in volatility estimation.
3. Implementation Methodology
3.1 Data Processing

    Market Data: Collected from Yahoo Finance for major financial indices and commodities.
    Feature Engineering:
        Log Returns computed as:
        rt=ln⁡(Pt/Pt−1)
        rt​=ln(Pt​/Pt−1​)
        Squared Returns used as input to model volatility patterns.
        VaR and ES "True" Labels generated using a GARCH(1,1) model with a Student’s t-distribution.
    Rolling Window: A 512-day rolling window is applied for estimating risk measures.

3.2 Model Architecture

We implemented SRNN-VE-3, the best-performing model from the paper, with:

    Recurrent Layer: Standard RNN with hidden size = 10.
    Hybrid Transformation:
        One branch uses raw hidden states.
        Another applies a nonlinear square-root transformation.
        The two outputs are summed and passed through a fully connected layer.
    Loss Function: Custom FZ0 loss function, designed for optimizing risk estimates.

3.3 Loss Function: FZ0

The FZ0 loss function is used to align the network outputs with regulatory backtesting requirements. It is defined as:
L=−1ES(1α(y⋅1y≤VaR)−ES)+VaRES+log⁡(−ES)−1
L=−ES1​(α1​(y⋅1y≤VaR​)−ES)+ESVaR​+log(−ES)−1

where:

    yy = True loss realization
    VaRVaR = Predicted VaR
    ESES = Predicted Expected Shortfall

To improve model robustness, we added:

    Deviation penalty to prevent underestimation:
    0.5⋅MSE(VaRpred,VaRtrue)
    0.5⋅MSE(VaRpred​,VaRtrue​)
    Confidence penalty to correct overconfidence in low-risk scenarios:
    0.1⋅∣VaRpred−VaRtrue∣
    0.1⋅∣VaRpred​−VaRtrue​∣

4. Training and Evaluation
4.1 Training

    The SRNN-VE-3 model is trained for each financial asset separately.
    Adam optimizer is used with:
        Learning rate = 10−410−4
        Weight decay = 10−310−3
        Gradient Clipping = 0.50.5
    Dropout (0.2) is applied for regularization.

4.2 Evaluation Metrics

    Prediction Error Distributions: Compare true vs. predicted risk measures.
    Scatter Plots: Show alignment between predicted VaR/ES and true values.
    Time-Series Analysis: Visualizes rolling predictions vs. actual VaR/ES.

5. Results and Observations
5.1 Prediction Accuracy

    SRNN-VE-3 underestimates extreme risk events, but its predictions align reasonably in normal volatility periods.
    Prediction error increases during financial crises, suggesting the need for further adjustments.

5.2 Model Limitations

    The R² values remain low, indicating the model does not fully explain market risk.
    The model struggles with extreme market movements, likely due to the limitations of squared returns as an input feature.

5.3 Possible Improvements

    Incorporate exogenous market variables (e.g., interest rates, macroeconomic indicators).
    Use alternative loss functions, such as the pinball loss, to better capture tail risk.
    Implement Bayesian recurrent models for better uncertainty quantification.

6. Conclusion

This project successfully implements the SRNN-based VaR and ES forecasting framework proposed in the paper. The results indicate that deep learning-based volatility modeling is feasible but requires further improvements for capturing extreme risk events.

7. References

Qiu, Z., Lazar, E., & Nakata, K. (2024). VaR and ES forecasting via recurrent neural network-based stateful models.
International Review of Financial Analysis, Volume 92, 103102.
https://doi.org/10.1016/j.irfa.2024.103102
(Full Paper)

Basel Committee on Banking Supervision. (2019). Minimum capital requirements for market risk.
https://www.bis.org/bcbs/publ/d457.pdf