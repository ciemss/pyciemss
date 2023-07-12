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
% Model name = Restif2006 - Whooping cough
%
% is http://identifiers.org/biomodels.db/MODEL1003290000
% is http://identifiers.org/biomodels.db/BIOMD0000000249
% isDescribedBy http://identifiers.org/pubmed/16615206
% isDerivedFrom http://identifiers.org/pubmed/460412
% isDerivedFrom http://identifiers.org/pubmed/460424
%


function main()
%Initial conditions vector
	x0=zeros(9,1);
	x0(1) = 1.0;
	x0(2) = 0.0588912;
	x0(3) = 0.003775;
	x0(4) = 1.0E-6;
	x0(5) = 0.93733;
	x0(6) = 0.0;
	x0(7) = 0.0;
	x0(8) = 0.0;
	x0(9) = 0.0;


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
% Parameter:   id =  beta_1, name = beta_1
% Parameter:   id =  R0_1, name = R0_1
	global_par_R0_1=17.0;
% Parameter:   id =  gamma_1, name = gamma_1
% Parameter:   id =  beta_2, name = beta_2
% Parameter:   id =  R0_2, name = R0_2
	global_par_R0_2=17.0;
% Parameter:   id =  gamma_2, name = gamma_2
% Parameter:   id =  tInf_1, name = infectious period 1
	global_par_tInf_1=21.0;
% Parameter:   id =  tInf_2, name = infectious period 2
	global_par_tInf_2=21.0;
% Parameter:   id =  sigma, name = sigma
% Parameter:   id =  tImm, name = immune period
	global_par_tImm=20.0;
% Parameter:   id =  Lambda_1, name = Lambda_1
% Parameter:   id =  Lambda_2, name = Lambda_2
% Parameter:   id =  I_1_frac, name = I_1_frac
% Parameter:   id =  I_2_frac, name = I_2_frac
% Parameter:   id =  S_frac, name = S_frac
% Parameter:   id =  R1_frac, name = R1_frac
% Parameter:   id =  R2_frac, name = R2_frac
% Parameter:   id =  Rp_frac, name = Rp_frac
% Parameter:   id =  psi, name = psi
	global_par_psi=0.2;
% assignmentRule: variable = mu
	global_par_mu=1/global_par_l_e;
% assignmentRule: variable = beta_1
	global_par_beta_1=global_par_R0_1*global_par_gamma_1;
% assignmentRule: variable = gamma_1
	global_par_gamma_1=365/global_par_tInf_1;
% assignmentRule: variable = beta_2
	global_par_beta_2=global_par_R0_2*global_par_gamma_2;
% assignmentRule: variable = gamma_2
	global_par_gamma_2=365/global_par_tInf_2;
% assignmentRule: variable = sigma
	global_par_sigma=1/global_par_tImm;
% assignmentRule: variable = Lambda_1
	global_par_Lambda_1=global_par_beta_1*(x(3)+x(7))/x(1);
% assignmentRule: variable = Lambda_2
	global_par_Lambda_2=global_par_beta_2*(x(4)+x(8))/x(1);
% assignmentRule: variable = I_1_frac
	global_par_I_1_frac=(x(3)+x(7))/x(1);
% assignmentRule: variable = I_2_frac
	global_par_I_2_frac=(x(4)+x(8))/x(1);
% assignmentRule: variable = S_frac
	global_par_S_frac=x(2)/x(1);
% assignmentRule: variable = R1_frac
	global_par_R1_frac=(x(5)+x(9))/x(1);
% assignmentRule: variable = R2_frac
	global_par_R2_frac=(x(6)+x(9))/x(1);
% assignmentRule: variable = Rp_frac
	global_par_Rp_frac=x(9)/x(1);

% Reaction: id = r1, name = Birth
	reaction_r1=global_par_mu*x(1);

% Reaction: id = r2, name = Death in S
	reaction_r2=global_par_mu*x(2);

% Reaction: id = r3, name = Death in I_1
	reaction_r3=global_par_mu*x(3);

% Reaction: id = r4, name = Death in I_2
	reaction_r4=global_par_mu*x(4);

% Reaction: id = r5, name = Death in R_1
	reaction_r5=global_par_mu*x(5);

% Reaction: id = r6, name = Death in R_2
	reaction_r6=global_par_mu*x(6);

% Reaction: id = r7, name = Death in I_1p
	reaction_r7=global_par_mu*x(7);

% Reaction: id = r8, name = Death in I_2p
	reaction_r8=global_par_mu*x(8);

% Reaction: id = r9, name = Death in R_p
	reaction_r9=global_par_mu*x(9);

% Reaction: id = r10, name = Primary Infection with strain 1
	reaction_r10=global_par_beta_1*(x(3)+x(7))/x(1)*x(2);

% Reaction: id = r11, name = Primary Infection with strain 2
	reaction_r11=global_par_beta_2*(x(4)+x(8))/x(1)*x(2);

% Reaction: id = r12, name = Secondary Infection with strain 1
	reaction_r12=(1-global_par_psi)*global_par_beta_1*(x(3)+x(7))/x(1)*x(6);

% Reaction: id = r13, name = Secondary Infection with strain 2
	reaction_r13=(1-global_par_psi)*global_par_beta_2*(x(4)+x(8))/x(1)*x(5);

% Reaction: id = r14, name = Recovery (I_1)
	reaction_r14=global_par_gamma_1*x(3);

% Reaction: id = r15, name = Recovery (I_2)
	reaction_r15=global_par_gamma_2*x(4);

% Reaction: id = r16, name = Recovery (I_1p)
	reaction_r16=global_par_gamma_1*x(7);

% Reaction: id = r17, name = Recovery (I_2p)
	reaction_r17=global_par_gamma_2*x(8);

% Reaction: id = r18, name = Loss of Immunity (R_1)
	reaction_r18=global_par_sigma*x(5);

% Reaction: id = r19, name = Loss of Immunity (R_2)
	reaction_r19=global_par_sigma*x(6);

% Reaction: id = r20, name = Loss of Immunity (R_p)
	reaction_r20=global_par_sigma*x(9);

	xdot=zeros(9,1);
	
% Species:   id = N, name = N
% Warning species is not changed by either rules or reactions
	xdot(1) = ;
	
% Species:   id = S, name = S, affected by kineticLaw
	xdot(2) = ( 1.0 * reaction_r1) + (-1.0 * reaction_r2) + (-1.0 * reaction_r10) + (-1.0 * reaction_r11) + ( 1.0 * reaction_r18) + ( 1.0 * reaction_r19) + ( 1.0 * reaction_r20);
	
% Species:   id = I_1, name = I_1, affected by kineticLaw
	xdot(3) = (-1.0 * reaction_r3) + ( 1.0 * reaction_r10) + (-1.0 * reaction_r14);
	
% Species:   id = I_2, name = I_2, affected by kineticLaw
	xdot(4) = (-1.0 * reaction_r4) + ( 1.0 * reaction_r11) + (-1.0 * reaction_r15);
	
% Species:   id = R_1, name = R_1, affected by kineticLaw
	xdot(5) = (-1.0 * reaction_r5) + (-1.0 * reaction_r13) + ( 1.0 * reaction_r14) + (-1.0 * reaction_r18);
	
% Species:   id = R_2, name = R_2, affected by kineticLaw
	xdot(6) = (-1.0 * reaction_r6) + (-1.0 * reaction_r12) + ( 1.0 * reaction_r15) + (-1.0 * reaction_r19);
	
% Species:   id = I_1p, name = I_1p, affected by kineticLaw
	xdot(7) = (-1.0 * reaction_r7) + ( 1.0 * reaction_r12) + (-1.0 * reaction_r16);
	
% Species:   id = I_2p, name = I_2p, affected by kineticLaw
	xdot(8) = (-1.0 * reaction_r8) + ( 1.0 * reaction_r13) + (-1.0 * reaction_r17);
	
% Species:   id = R_p, name = R_p, affected by kineticLaw
	xdot(9) = (-1.0 * reaction_r9) + ( 1.0 * reaction_r16) + ( 1.0 * reaction_r17) + (-1.0 * reaction_r20);
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


