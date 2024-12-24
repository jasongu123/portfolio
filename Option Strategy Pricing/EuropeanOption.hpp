#include <string>
using std::string;
double N(double x);
class EuropeanOption {
private:
      void init();       // Initialise all default values
      void copy(const EuropeanOption& o2);
      // "Kernel" functions for option calculations
      double CallPrice() const;
      double PutPrice() const;
      double CallDelta() const;
      double PutDelta() const;
      double Gamma() const;
      double CallTheta() const;
      double PutTheta() const;
      double Vega() const;
public:
      // Public member data for convenience only
double r;
double sig;
double K;
double T;
double U;
double b;
// Interest rate
// Volatility
// Strike price
// Expiry date
// Current underlying price
// Cost of carry
string optType; // Option name (call, put)
public:
// Constructors
EuropeanOption(); // Default call option
EuropeanOption(const EuropeanOption& option2); // Copy constructor
EuropeanOption (const string& optionType);    // Create option type
// Destructor
virtual ~EuropeanOption();
// Assignment operator
EuropeanOption& operator = (const EuropeanOption& option2);
// Functions that calculate option price and (some) sensitivities
double Price() const;
double Delta() const;
double getGamma() const;
double Theta() const;
double getVega() const;
// Modifier functions
void toggle();           // Change option type (C/P, P/C)
};