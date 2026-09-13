#pragma once

#include <string>

double N(double x);
double normalPdf(double x);

struct ImpliedVolatilityResult {
      bool success;
      double volatility;
      int iterations;
      std::string error;
};

class EuropeanOption {
private:
      void init();
      void copy(const EuropeanOption& option);
      double intrinsicCall() const;
      double intrinsicPut() const;
      double discountedSpot() const;
      double discountedStrike() const;
      double forwardIntrinsicCall() const;
      double forwardIntrinsicPut() const;
      double PriceAtVolatility(double volatility) const;
      double PriceAtSpot(double spot) const;
      double PriceAtExpiry(double expiry) const;
      double AmericanPrice() const;
      double d1() const;
      double d2() const;
      double CallPrice() const;
      double PutPrice() const;
      double CallDelta() const;
      double PutDelta() const;
      double Gamma() const;
      double CallTheta() const;
      double PutTheta() const;
      double Vega() const;
public:
      // Public member data for convenience only.
      double r;
      double sig;
      double K;
      double T;
      double U;
      double b;
      int binomialSteps;
      std::string optType;
      std::string exerciseStyle;
public:
      EuropeanOption();
      EuropeanOption(const EuropeanOption& option2);
      explicit EuropeanOption(const std::string& optionType);
      virtual ~EuropeanOption();

      EuropeanOption& operator = (const EuropeanOption& option2);

      double Price() const;
      double Delta() const;
      double getGamma() const;
      double Theta() const;
      double getVega() const;
      double MinimumArbitragePrice() const;
      double MaximumArbitragePrice() const;
      ImpliedVolatilityResult SolveImpliedVolatility(double marketPrice,
                                                     double tolerance = 1e-8,
                                                     int maxIterations = 200) const;

      bool IsCall() const;
      bool IsPut() const;
      bool IsEuropean() const;
      bool IsAmerican() const;
      bool HasValidParameters() const;
      std::string ValidationError() const;
      void SetOptionType(const std::string& optionType);
      void SetExerciseStyle(const std::string& style);
      void toggle();
};
