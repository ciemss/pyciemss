% This file works with OCTAVE and is automatically generated with 
% the System Biology Format Converter (http://sbfc.sourceforge.net/)
% from an SBML file.
% To run this file with Matlab you must edit the comments providing
% the definition of the ode solver and the signature for the 
% xdot function.
%
% The conversion system has the following limitations:
%  - You may have to re order some reactions and Assignment Rules definition
%  - Delays are not taken into account
%  - You should change the lsode parameters (start, end, steps) to get better results
%

%
% Model name = Restif2007 - Vaccination invasion
%
% is http://identifiers.org/biomodels.db/MODEL1012210000
% is http://identifiers.org/biomodels.db/BIOMD0000000294
% isDescribedBy http://identifiers.org/pubmed/17210532
% isDerivedFrom http://identifiers.org/biomodels.db/BIOMD0000000249
% isDerivedFrom http://identifiers.org/pubmed/460412
% isDerivedFrom http://identifiers.org/pubmed/460424
%


function main()
%Initial conditions vector
	x0=zeros(11,1);
	x0(1) = 1.0;
	x0(2) = 0.0588235;
	x0(3) = 0.00176967;
	x0(4) = 1.0E-6;
	x0(5) = 0.439407;
	x0(6) = 0.0;
	x0(7) = 0.9;
	x0(8) = 0.5;
	x0(9) = 0.0;
	x0(10) = 0.0;
	x0(11) = 0.0;


% Depending on whether you are using Octave or Matlab,
% you should comment / uncomment one of the following blocks.
% This should also be done for the definition of the function f below.
% Start Matlab code
%	tspan=[0:0.01:100];
%	opts = odeset('AbsTol',1e-3);
%	[t,x]=ode23tb(@f,tspan,x0,opts);
% End Matlab code

% Start Octave code
	t=linspace(0,100,100);
	x=lsode('f',x0,t);
% End Octave code


	plot(t,x);
end



% Depending on whether you are using Octave or Matlab,
% you should comment / uncomment one of the following blocks.
% This should also be done for the definition of the function f below.
% Start Matlab code
%function xdot=f(t,x)
% End Matlab code

% Start Octave code
function xdot=f(x,t)
% End Octave code

% Compartment: id = env, name = environment, constant
	compartment_env=1.0;
% Parameter:   id =  mu, name = mu
% Parameter:   id =  l_e, name = life expectancy
	global_par_l_e=50.0;
% Parameter:   id =  beta, name = beta
% Parameter:   id =  R0, name = R0
	global_par_R0=17.0;
% Parameter:   id =  gamma, name = gamma
% Parameter:   id =  p, name = p
	global_par_p=1.0;
% Parameter:   id =  tau, name = tau
	global_par_tau=0.7;
% Parameter:   id =  theta, name = theta
	global_par_theta=0.5;
% Parameter:   id =  nu, name = nu
	global_par_nu=0.5;
% Parameter:   id =  eta, name = eta
	global_par_eta=0.5;
% Parameter:   id =  sigma, name = sigma
% Parameter:   id =  sigmaV, name = sigmaV
% Parameter:   id =  tInf, name = infectious period (d)
	global_par_tInf=21.0;
% Parameter:   id =  tImm, name = immune period (yr)
	global_par_tImm=20.0;
% Parameter:   id =  tImm_V, name = vaccine immune period (yr)
	global_par_tImm_V=50.0;
% Parameter:   id =  strain1_frac, name = strain1_frac
% Parameter:   id =  strain2_frac, name = strain2_frac
% Parameter:   id =  S_frac, name = S_frac
% Parameter:   id =  V_frac, name = V_frac
% Parameter:   id =  R_1_frac, name = R_1_frac
% Parameter:   id =  R_2_frac, name = R_2_frac
% Parameter:   id =  R_frac, name = R_frac
% assignmentRule: variable = mu
	global_par_mu=1/global_par_l_e;
% assignmentRule: variable = beta
	global_par_beta=global_par_R0*(global_par_gamma+global_par_mu);
% assignmentRule: variable = gamma
	global_par_gamma=365/global_par_tInf;
% assignmentRule: variable = sigma
	global_par_sigma=1/global_par_tImm;
% assignmentRule: variable = sigmaV
	global_par_sigmaV=1/global_par_tImm_V;
% assignmentRule: variable = strain1_frac
	global_par_strain1_frac=(x(3)+x(10))/x(1);
% assignmentRule: variable = strain2_frac
	global_par_strain2_frac=(x(4)+x(9)+x(8))/x(1);
% assignmentRule: variable = S_frac
	global_par_S_frac=x(2)/x(1);
% assignmentRule: variable = V_frac
	global_par_V_frac=x(7)/x(1);
% assignmentRule: variable = R_1_frac
	global_par_R_1_frac=(x(5)+x(11))/x(1);
% assignmentRule: variable = R_2_frac
	global_par_R_2_frac=(x(6)+x(11))/x(1);
% assignmentRule: variable = R_frac
	global_par_R_frac=x(11)/x(1);

% Reaction: id = r1, name = Birth S (unvaccinated)
	reaction_r1=global_par_mu*(1-global_par_p)*x(1);

% Reaction: id = r2, name = Birth V (vaccinated)
	reaction_r2=global_par_mu*global_par_p*x(1);

% Reaction: id = r3, name = Death in S
	reaction_r3=global_par_mu*x(2);

% Reaction: id = r4, name = Death in V
	reaction_r4=global_par_mu*x(7);

% Reaction: id = r5, name = Death in I1
	reaction_r5=global_par_mu*x(3);

% Reaction: id = r6, name = Death in I2
	reaction_r6=global_par_mu*x(4);

% Reaction: id = r7, name = Death in Iv2
	reaction_r7=global_par_mu*x(8);

% Reaction: id = r8, name = Death in R1
	reaction_r8=global_par_mu*x(5);

% Reaction: id = r9, name = Death in R2
	reaction_r9=global_par_mu*x(6);

% Reaction: id = r10, name = Death in J1
	reaction_r10=global_par_mu*x(10);

% Reaction: id = r11, name = Death in J2
	reaction_r11=global_par_mu*x(9);

% Reaction: id = r12, name = Death in Rp
	reaction_r12=global_par_mu*x(11);

% Reaction: id = r13, name = Primary Infection with strain 1
	reaction_r13=global_par_beta*x(2)*(x(3)+x(10))/x(1);

% Reaction: id = r14, name = Primary Infection with strain 2
	reaction_r14=global_par_beta*x(2)*(x(4)+x(9)+x(8))/x(1);

% Reaction: id = r15, name = Primary Infection of V with strain 2
	reaction_r15=global_par_beta*(1-global_par_tau)*x(7)*(x(4)+x(9)+x(8))/x(1);

% Reaction: id = r16, name = Recovery (I1)
	reaction_r16=global_par_gamma*x(3);

% Reaction: id = r17, name = Recovery (I2)
	reaction_r17=global_par_gamma*x(4);

% Reaction: id = r18, name = Secondary Infection with strain 1
	reaction_r18=global_par_beta*(1-global_par_theta)*x(6)*(x(3)+x(10))/x(1);

% Reaction: id = r19, name = Secondary Infection with strain 2
	reaction_r19=global_par_beta*(1-global_par_theta)*x(5)*(x(4)+x(9)+x(8))/x(1);

% Reaction: id = r20, name = Recovery (J1)
	reaction_r20=global_par_gamma/(1-global_par_nu)*x(10);

% Reaction: id = r21, name = Recovery (J2)
	reaction_r21=global_par_gamma/(1-global_par_nu)*x(9);

% Reaction: id = r22, name = Recovery (Iv2)
	reaction_r22=global_par_gamma/(1-global_par_eta)*x(8);

% Reaction: id = r23, name = Loss of Immunity (R1)
	reaction_r23=global_par_sigma*x(5);

% Reaction: id = r24, name = Loss of Immunity (R2)
	reaction_r24=global_par_sigma*x(6);

% Reaction: id = r25, name = Loss of Immunity (Rp)
	reaction_r25=global_par_sigma*x(11);

% Reaction: id = r26, name = Loss of Immunity (V)
	reaction_r26=global_par_sigmaV*x(7);

	xdot=zeros(11,1);
	
% Species:   id = N, name = N
% Warning species is not changed by either rules or reactions
	xdot(1) = ;
	
% Species:   id = S, name = S, affected by kineticLaw
	xdot(2) = ( 1.0 * reaction_r1) + (-1.0 * reaction_r3) + (-1.0 * reaction_r13) + (-1.0 * reaction_r14) + ( 1.0 * reaction_r23) + ( 1.0 * reaction_r24) + ( 1.0 * reaction_r25) + ( 1.0 * reaction_r26);
	
% Species:   id = I1, name = I1, affected by kineticLaw
	xdot(3) = (-1.0 * reaction_r5) + ( 1.0 * reaction_r13) + (-1.0 * reaction_r16);
	
% Species:   id = I2, name = I2, affected by kineticLaw
	xdot(4) = (-1.0 * reaction_r6) + ( 1.0 * reaction_r14) + (-1.0 * reaction_r17);
	
% Species:   id = R1, name = R1, affected by kineticLaw
	xdot(5) = (-1.0 * reaction_r8) + ( 1.0 * reaction_r16) + (-1.0 * reaction_r19) + (-1.0 * reaction_r23);
	
% Species:   id = R2, name = R2, affected by kineticLaw
	xdot(6) = (-1.0 * reaction_r9) + ( 1.0 * reaction_r17) + (-1.0 * reaction_r18) + (-1.0 * reaction_r24);
	
% Species:   id = V, name = V, affected by kineticLaw
	xdot(7) = ( 1.0 * reaction_r2) + (-1.0 * reaction_r4) + (-1.0 * reaction_r15) + (-1.0 * reaction_r26);
	
% Species:   id = Iv2, name = Iv2, affected by kineticLaw
	xdot(8) = (-1.0 * reaction_r7) + ( 1.0 * reaction_r15) + (-1.0 * reaction_r22);
	
% Species:   id = J2, name = J2, affected by kineticLaw
	xdot(9) = (-1.0 * reaction_r11) + ( 1.0 * reaction_r19) + (-1.0 * reaction_r21);
	
% Species:   id = J1, name = J1, affected by kineticLaw
	xdot(10) = (-1.0 * reaction_r10) + ( 1.0 * reaction_r18) + (-1.0 * reaction_r20);
	
% Species:   id = R, name = R, affected by kineticLaw
	xdot(11) = (-1.0 * reaction_r12) + ( 1.0 * reaction_r20) + ( 1.0 * reaction_r21) + ( 1.0 * reaction_r22) + (-1.0 * reaction_r25);
end

% adding few functions representing operators used in SBML but not present directly 
% in either matlab or octave. 
function z=pow(x,y),z=x^y;end
function z=root(x,y),z=y^(1/x);end
function z = piecewise(varargin)
	numArgs = nargin;
	result = 0;
	foundResult = 0;
	for k=1:2: numArgs-1
		if varargin{k+1} == 1
			result = varargin{k};
			foundResult = 1;
			break;
		end
	end
	if foundResult == 0
		result = varargin{numArgs};
	end
	z = result;
end


