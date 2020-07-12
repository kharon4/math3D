#pragma once

#include<vector>
#include<iostream>

namespace LES {//Linear equation solver


	class eqn {
	private:
		std::vector<double> coefficients;//coefficients for the variables
		double constant;//to the right of the equal sign
	public:

		eqn(unsigned long int noCoefficients = 0, double* coeff = nullptr, double Constant = 0);

		//returns no of coefficients
		inline unsigned long int noCoeff() const; 
		
		//changes no of coefficients , values of cofficients after the function call are not predictable
		inline void setNoCoeff(unsigned long int no);


		//if index < noCoefficients, the coefficients r being reffered to, if index==noCoefficients constant is being reffered to
		double getCoeff(unsigned long int index) const;
		bool setCoeff(unsigned long int index, double val);

		inline double getConst() const;
		inline void setConst(double val);

		void displayEqn(std::ostream& f, bool showVar = false);
	};


	class system {
	public:
		enum solType : unsigned char { uniqueSol = 0, inconsistant = 1, infiniteSols = 2 };
		enum sysType : unsigned char { criticallyDefined = 0, underDefined = 1, overDefined = 2 };
	private:
		solType solutionType;
		sysType systemType;
		eqn* eqns = nullptr;
		unsigned long int noEqns = 0;
		unsigned long int noCoef = 0;

		void calculateSysType();

	public:

		std::vector<double> sols;

		system(unsigned long int noCoefficients, unsigned long int noEquations);
		~system();

		inline unsigned long int getNoCoeff() const;
		inline unsigned long int getNoEqns() const;

		inline void changeNoCoeff(unsigned long int noC);
		inline void changeNoEqns(unsigned long int noE);
		inline void changeSystemSize(unsigned long int noC, unsigned long int noE);

		void displayMatrix(std::ostream& f, bool showVar = false);

		bool load(std::istream& f);

		void solve();
		
	
	};
}
