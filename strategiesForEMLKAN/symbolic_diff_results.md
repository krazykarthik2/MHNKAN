# EML-KAN Symbolic Decomposition & Differentiation (Pure AdamW)

This report presents the symbolic decomposition, learned equations, solved forms, and analytical derivatives for target functions fitted using EML-KAN under pure AdamW optimization (no L-BFGS fine-tuning).

## 1. Summary Comparison Table

| Target Function | Learned Equation | Target Derivative | Learned Derivative | Fit MSE Loss |
| :--- | :--- | :--- | :--- | :--- |
| $f_1(x) = e^{2x} - \ln(\text{softplus}(x) + 10^{-6})$ | $- 0.361948314876303 x + 0.977084473209322 e^{2.03033215335993 x} - 0.554403539269242 \log{\left(\log{\left(0.630904922183483 e^{0.835557023393546 x} + 1 \right)} + 1.0 \cdot 10^{-6} \right)}$ | $2 e^{2 x} - \frac{e^{x}}{\left(e^{x} + 1\right) \left(\log{\left(e^{x} + 1 \right)} + 1.0 \cdot 10^{-6}\right)}$ | $1.98380602250563 e^{2.03033215335993 x} - 0.361948314876303 - \frac{0.2922577280747 e^{0.835557023393546 x}}{\left(0.630904922183483 e^{0.835557023393546 x} + 1\right) \left(\log{\left(0.630904922183483 e^{0.835557023393546 x} + 1 \right)} + 1.0 \cdot 10^{-6}\right)}$ | 0.00005846 |
| $f_2(x) = e^{-1.5x} - \ln(\text{softplus}(0.5x+0.2) + 10^{-6})$ | $- 0.265565186457942 x - 0.523975795609931 \log{\left(\log{\left(0.775773089851936 e^{0.286465432383557 x} + 1 \right)} + 1.0 \cdot 10^{-6} \right)} + 0.938888073288872 e^{- 1.54673341517036 x}$ | $- 1.5 e^{- 1.5 x} - \frac{0.610701379080085 e^{0.5 x}}{\left(1.22140275816017 e^{0.5 x} + 1\right) \left(\log{\left(1.22140275816017 e^{0.5 x} + 1 \right)} + 1.0 \cdot 10^{-6}\right)}$ | $-0.265565186457942 - 1.45220955606081 e^{- 1.54673341517036 x} - \frac{0.116444279980549 e^{0.286465432383557 x}}{\left(0.775773089851936 e^{0.286465432383557 x} + 1\right) \left(\log{\left(0.775773089851936 e^{0.286465432383557 x} + 1 \right)} + 1.0 \cdot 10^{-6}\right)}$ | 0.00001933 |
| $f_3(x) = 0.5x + 1.2 [e^{0.8x} - \ln(\text{softplus}(x) + 10^{-6})]$ | $0.915667988121653 x - 0.611297155121402 \log{\left(\log{\left(1 + 0.640649933489212 e^{- 1.54038885909345 x} \right)} + 1.0 \cdot 10^{-6} \right)} + 1.220839538708 e^{- 0.764072133391393 x}$ | $0.96 e^{0.8 x} + 0.5 - \frac{1.2 e^{x}}{\left(e^{x} + 1\right) \left(\log{\left(e^{x} + 1 \right)} + 1.0 \cdot 10^{-6}\right)}$ | $0.915667988121653 - 0.932809470869185 e^{- 0.764072133391393 x} + \frac{0.603258609834365 e^{- 1.54038885909345 x}}{\left(1 + 0.640649933489212 e^{- 1.54038885909345 x}\right) \left(\log{\left(1 + 0.640649933489212 e^{- 1.54038885909345 x} \right)} + 1.0 \cdot 10^{-6}\right)}$ | 0.00113775 |

## 2. Parameter Details

### Function 1 Parameters:
- $w_{base} = -0.361948$
- $w_{eml} = 0.554404$
- $a = 2.030332$
- $b = 0.566680$
- $c = 0.835557$
- $d = -0.460600$

### Function 2 Parameters:
- $w_{base} = -0.265565$
- $w_{eml} = 0.523976$
- $a = -1.546733$
- $b = 0.583251$
- $c = 0.286465$
- $d = -0.253895$

### Function 3 Parameters:
- $w_{base} = 0.915668$
- $w_{eml} = 0.611297$
- $a = -0.764072$
- $b = 0.691711$
- $c = -1.540389$
- $d = -0.445272$
