#include "EuropeanOption.hpp"
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr double kBusinessDaysPerYear = 252.0;
constexpr double kDefaultHighVolatility = 0.25;
constexpr double kMaximumSolverVolatility = 20.0;
constexpr double kMinimumSpotBump = 0.01;
constexpr double kRelativeSpotBump = 0.01;
constexpr double kVolatilityBump = 1e-3;

struct AmericanMetrics {
      double price;
      double delta;
      double gamma;
      double theta;
};

double intrinsicValue(const EuropeanOption& option, double spot)
{
      return option.IsCall() ? std::max(spot - option.K, 0.0)
                             : std::max(option.K - spot, 0.0);
}

AmericanMetrics PriceAmericanCrr(const EuropeanOption& option)
{
      if (option.T <= 0.0 || option.U <= 0.0) {
            return {intrinsicValue(option, option.U), 0.0, 0.0, 0.0};
      }

      if (option.sig <= 0.0) {
            const int steps = std::max(option.binomialSteps, 1);
            double bestValue = 0.0;
            for (int i = 0; i <= steps; ++i) {
                  const double t = option.T * static_cast<double>(i) / static_cast<double>(steps);
                  const double spot = option.U * std::exp(option.b * t);
                  bestValue = std::max(
                      bestValue,
                      std::exp(-option.r * t) * intrinsicValue(option, spot));
            }
            return {bestValue, 0.0, 0.0, 0.0};
      }

      int steps = std::max(option.binomialSteps, 3);
      double dt = option.T / static_cast<double>(steps);
      double up = 0.0;
      double down = 0.0;
      double probability = 0.0;
      while (true) {
            dt = option.T / static_cast<double>(steps);
            up = std::exp(option.sig * std::sqrt(dt));
            down = 1.0 / up;
            probability = (std::exp(option.b * dt) - down) / (up - down);
            if (probability > 0.0 && probability < 1.0) {
                  break;
            }
            if (steps >= 20000) {
                  const double intrinsic = intrinsicValue(option, option.U);
                  return {intrinsic, 0.0, 0.0, 0.0};
            }
            steps *= 2;
      }

      const double discount = std::exp(-option.r * dt);
      std::vector<double> optionValues(static_cast<std::size_t>(steps + 1), 0.0);
      for (int j = 0; j <= steps; ++j) {
            const double spot = option.U * std::pow(up, j) * std::pow(down, steps - j);
            optionValues[static_cast<std::size_t>(j)] = intrinsicValue(option, spot);
      }

      double valueDown = 0.0;
      double valueUp = 0.0;
      double valueDownDown = 0.0;
      double valueUpDown = 0.0;
      double valueUpUp = 0.0;

      for (int i = steps - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                  const double continuation =
                      discount * (probability * optionValues[static_cast<std::size_t>(j + 1)] +
                                  (1.0 - probability) * optionValues[static_cast<std::size_t>(j)]);
                  const double spot = option.U * std::pow(up, j) * std::pow(down, i - j);
                  optionValues[static_cast<std::size_t>(j)] =
                      std::max(continuation, intrinsicValue(option, spot));
            }

            if (i == 2) {
                  valueDownDown = optionValues[0];
                  valueUpDown = optionValues[1];
                  valueUpUp = optionValues[2];
            } else if (i == 1) {
                  valueDown = optionValues[0];
                  valueUp = optionValues[1];
            }
      }

      const double price = optionValues[0];
      const double upSpot = option.U * up;
      const double downSpot = option.U * down;
      const double delta = (valueUp - valueDown) / (upSpot - downSpot);

      const double upUpSpot = option.U * up * up;
      const double downDownSpot = option.U * down * down;
      const double deltaUp = (valueUpUp - valueUpDown) / (upUpSpot - option.U);
      const double deltaDown = (valueUpDown - valueDownDown) / (option.U - downDownSpot);
      const double gamma = (deltaUp - deltaDown) / (0.5 * (upUpSpot - downDownSpot));
      const double theta = (valueUpDown - price) / (2.0 * dt) / kBusinessDaysPerYear;

      return {price, delta, gamma, theta};
}
}

double N(double x) {
    return 0.5 * (1 + std::erf(x / std::sqrt(2)));
}

double normalPdf(double x)
{
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::acos(-1.0));
}

double EuropeanOption::intrinsicCall() const
{
      return std::max(U - K, 0.0);
}

double EuropeanOption::intrinsicPut() const
{
      return std::max(K - U, 0.0);
}

double EuropeanOption::discountedSpot() const
{
      return U * std::exp((b - r) * T);
}

double EuropeanOption::discountedStrike() const
{
      return K * std::exp(-r * T);
}

double EuropeanOption::forwardIntrinsicCall() const
{
      return std::max(discountedSpot() - discountedStrike(), 0.0);
}

double EuropeanOption::forwardIntrinsicPut() const
{
      return std::max(discountedStrike() - discountedSpot(), 0.0);
}

double EuropeanOption::PriceAtVolatility(double volatility) const
{
      EuropeanOption option(*this);
      option.sig = volatility;
      return option.Price();
}

double EuropeanOption::PriceAtSpot(double spot) const
{
      EuropeanOption option(*this);
      option.U = spot;
      return option.Price();
}

double EuropeanOption::PriceAtExpiry(double expiry) const
{
      EuropeanOption option(*this);
      option.T = std::max(expiry, 0.0);
      return option.Price();
}

double EuropeanOption::AmericanPrice() const
{
      return PriceAmericanCrr(*this).price;
}

double EuropeanOption::d1() const
{
      const double sigmaRootT = sig * std::sqrt(T);
      return (std::log(U / K) + (b + 0.5 * sig * sig) * T) / sigmaRootT;
}

double EuropeanOption::d2() const
{
      return d1() - sig * std::sqrt(T);
}

double EuropeanOption::CallPrice() const
{
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return forwardIntrinsicCall();
      }

      const double optionD1 = d1();
      const double optionD2 = d2();
      return discountedSpot() * N(optionD1) - discountedStrike() * N(optionD2);
}

double EuropeanOption::PutPrice() const
{
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return forwardIntrinsicPut();
      }

      const double optionD1 = d1();
      const double optionD2 = d2();
      return discountedStrike() * N(-optionD2) - discountedSpot() * N(-optionD1);
}

double EuropeanOption::CallDelta() const
{
      const double carryFactor = std::exp((b - r) * T);
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return intrinsicCall() > 0.0 ? carryFactor : 0.0;
      }

      return carryFactor * N(d1());
}

double EuropeanOption::PutDelta() const
{
      const double carryFactor = std::exp((b - r) * T);
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return intrinsicPut() > 0.0 ? -carryFactor : 0.0;
      }

      return carryFactor * (N(d1()) - 1.0);
}

double EuropeanOption::Gamma() const
{
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return 0.0;
      }

      return std::exp((b - r) * T) * normalPdf(d1()) / (U * sig * std::sqrt(T));
}

double EuropeanOption::CallTheta() const
{
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return 0.0;
      }

      const double optionD1 = d1();
      const double optionD2 = d2();
      const double carrySpot = discountedSpot();
      const double diffusion = -(carrySpot * normalPdf(optionD1) * sig) / (2.0 * std::sqrt(T));
      const double carry = -(b - r) * carrySpot * N(optionD1);
      const double discount = -r * discountedStrike() * N(optionD2);
      return (diffusion + carry + discount) / kBusinessDaysPerYear;
}

double EuropeanOption::PutTheta() const
{
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return 0.0;
      }

      const double optionD1 = d1();
      const double optionD2 = d2();
      const double carrySpot = discountedSpot();
      const double diffusion = -(carrySpot * normalPdf(optionD1) * sig) / (2.0 * std::sqrt(T));
      const double carry = (b - r) * carrySpot * N(-optionD1);
      const double discount = r * discountedStrike() * N(-optionD2);
      return (diffusion + carry + discount) / kBusinessDaysPerYear;
}

double EuropeanOption::Vega() const
{
      if (T <= 0.0 || sig <= 0.0 || U <= 0.0) {
            return 0.0;
      }

      return discountedSpot() * normalPdf(d1()) * std::sqrt(T);
}

void EuropeanOption::init()
{
      r = 0.05;
      sig = 0.30;
      K = 65.0;
      T = 0.25;
      U = 60.0;
      b = r;
      binomialSteps = 200;
      optType = "C";
      exerciseStyle = "E";
}

void EuropeanOption::copy(const EuropeanOption& o2)
{
      r = o2.r;
      sig = o2.sig;
      K = o2.K;
      T = o2.T;
      U = o2.U;
      b = o2.b;
      binomialSteps = o2.binomialSteps;
      optType = o2.optType;
      exerciseStyle = o2.exerciseStyle;
}

EuropeanOption::EuropeanOption()
{
      init();
}

EuropeanOption::EuropeanOption(const EuropeanOption& o2)
{
      copy(o2);
}

EuropeanOption::EuropeanOption(const std::string& optionType)
{
      init();
      SetOptionType(optionType);
}

EuropeanOption::~EuropeanOption()
{
}

EuropeanOption& EuropeanOption::operator = (const EuropeanOption& opt2)
{
      if (this == &opt2) {
            return *this;
      }

      copy(opt2);
      return *this;
}

double EuropeanOption::Price() const
{
      if (IsAmerican()) {
            return AmericanPrice();
      }

      if (IsCall()) {
            return CallPrice();
      }

      return PutPrice();
}

double EuropeanOption::Delta() const
{
      if (IsAmerican()) {
            return PriceAmericanCrr(*this).delta;
      }

      if (IsCall()) {
            return CallDelta();
      }

      return PutDelta();
}

double EuropeanOption::getGamma() const
{
      if (IsAmerican()) {
            return PriceAmericanCrr(*this).gamma;
      }

      return Gamma();
}

double EuropeanOption::Theta() const
{
      if (IsAmerican()) {
            return PriceAmericanCrr(*this).theta;
      }

      if (IsCall()) {
            return CallTheta();
      }

      return PutTheta();
}

double EuropeanOption::getVega() const
{
      if (IsAmerican()) {
            const double upVolatility = sig + kVolatilityBump;
            const double downVolatility = std::max(sig - kVolatilityBump, 0.0);
            return (PriceAtVolatility(upVolatility) - PriceAtVolatility(downVolatility)) /
                   (upVolatility - downVolatility);
      }

      return Vega();
}

double EuropeanOption::MinimumArbitragePrice() const
{
      if (IsAmerican()) {
            return IsCall() ? intrinsicCall() : intrinsicPut();
      }

      if (IsCall()) {
            return forwardIntrinsicCall();
      }

      return forwardIntrinsicPut();
}

double EuropeanOption::MaximumArbitragePrice() const
{
      if (IsAmerican()) {
            return IsCall() ? U : K;
      }

      if (IsCall()) {
            return discountedSpot();
      }

      return discountedStrike();
}

ImpliedVolatilityResult EuropeanOption::SolveImpliedVolatility(double marketPrice,
                                                               double tolerance,
                                                               int maxIterations) const
{
      ImpliedVolatilityResult result{
          false,
          std::numeric_limits<double>::quiet_NaN(),
          0,
          ""
      };

      if (!IsCall() && !IsPut()) {
            result.error = "option type must be C or P";
            return result;
      }
      if (marketPrice < 0.0) {
            result.error = "market price must be non-negative";
            return result;
      }
      if (K <= 0.0) {
            result.error = "strike must be positive";
            return result;
      }
      if (U <= 0.0) {
            result.error = "spot must be positive";
            return result;
      }
      if (T <= 0.0) {
            result.error = "implied volatility is undefined at expiry";
            return result;
      }
      if (tolerance <= 0.0) {
            result.error = "tolerance must be positive";
            return result;
      }
      if (maxIterations <= 0) {
            result.error = "maxIterations must be positive";
            return result;
      }

      const double lowerBound = MinimumArbitragePrice();
      const double upperBound = MaximumArbitragePrice();
      const double boundTolerance = std::max(tolerance, 1e-10);
      if (marketPrice < lowerBound - boundTolerance || marketPrice > upperBound + boundTolerance) {
            result.error = "market price violates no-arbitrage bounds";
            return result;
      }
      if (std::abs(marketPrice - lowerBound) <= tolerance) {
            result.success = true;
            result.volatility = 0.0;
            return result;
      }

      double lowVolatility = 0.0;
      double highVolatility = std::max(sig, kDefaultHighVolatility);
      double highPrice = PriceAtVolatility(highVolatility);
      while (highPrice < marketPrice && highVolatility < kMaximumSolverVolatility) {
            highVolatility = std::min(highVolatility * 2.0, kMaximumSolverVolatility);
            highPrice = PriceAtVolatility(highVolatility);
      }

      if (highPrice < marketPrice) {
            result.error = "could not bracket the implied volatility";
            return result;
      }

      for (int iteration = 1; iteration <= maxIterations; ++iteration) {
            const double midVolatility = 0.5 * (lowVolatility + highVolatility);
            const double midPrice = PriceAtVolatility(midVolatility);
            const double priceError = midPrice - marketPrice;
            result.iterations = iteration;

            if (std::abs(priceError) <= tolerance || (highVolatility - lowVolatility) <= tolerance) {
                  result.success = true;
                  result.volatility = midVolatility;
                  result.error.clear();
                  return result;
            }

            if (priceError < 0.0) {
                  lowVolatility = midVolatility;
            } else {
                  highVolatility = midVolatility;
            }
      }

      result.volatility = 0.5 * (lowVolatility + highVolatility);
      result.error = "solver did not converge within the iteration limit";
      return result;
}

bool EuropeanOption::IsCall() const
{
      return optType == "C";
}

bool EuropeanOption::IsPut() const
{
      return optType == "P";
}

bool EuropeanOption::IsEuropean() const
{
      return exerciseStyle == "E";
}

bool EuropeanOption::IsAmerican() const
{
      return exerciseStyle == "A";
}

bool EuropeanOption::HasValidParameters() const
{
      return ValidationError().empty();
}

std::string EuropeanOption::ValidationError() const
{
      if (!IsCall() && !IsPut()) {
            return "option type must be C or P";
      }
      if (K <= 0.0) {
            return "strike must be positive";
      }
      if (U < 0.0) {
            return "spot must be non-negative";
      }
      if (T < 0.0) {
            return "expiry must be non-negative";
      }
      if (sig < 0.0) {
            return "volatility must be non-negative";
      }
      if (binomialSteps <= 0) {
            return "binomial steps must be positive";
      }
      if (!IsEuropean() && !IsAmerican()) {
            return "exercise style must be E or A";
      }

      return "";
}

void EuropeanOption::SetOptionType(const std::string& optionType)
{
      if (optionType == "c") {
            optType = "C";
      } else if (optionType == "p") {
            optType = "P";
      } else {
            optType = optionType;
      }
}

void EuropeanOption::SetExerciseStyle(const std::string& style)
{
      if (style == "e") {
            exerciseStyle = "E";
      } else if (style == "a") {
            exerciseStyle = "A";
      } else {
            exerciseStyle = style;
      }
}

void EuropeanOption::toggle()
{
      if (IsCall()) {
            optType = "P";
      } else {
            optType = "C";
      }
}
