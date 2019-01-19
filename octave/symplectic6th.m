%% 6th-order symplectic integrator for u'=Av, v'=Bu (H(u,v)=v'Av/2-u'Bu/2)
% optimized version; for a better illustration of the idea, see symplectic4th.m
clear;
clc;

%% parameters
A=1;
B=-1;
dim=size(A,2);
u0=1;    v0=0.3;        % NOTE: this can be changed
T=10000;
h=0.1;                 % try h=0.1 and h=0.001
coarse=10;             % store the result every $coarse$ steps
gamma2=1/(2-2^(1/3));
gamma4=1/(2-2^(1/5));

%% symplectic integration
TimeSpan=[0:h*coarse:T];   TimeSteps=length(TimeSpan)-1;
u=zeros(dim,TimeSteps+1);   u(:,1)=u0;
v=zeros(dim,TimeSteps+1);   v(:,1)=v0;
E=zeros(1,TimeSteps+1);
uu=u(:,1);  vv=v(:,1);
i=0;    E(i+1)=v(:,i+1)'*A*v(:,i+1)/2-u(:,i+1)'*B*u(:,i+1)/2;
uu=uu+gamma2*gamma4*h/2*A*vv;
for i=1:TimeSteps
    for ii=1:coarse
        vv=vv+gamma2*gamma4*h*B*uu;
        uu=uu+(1-gamma2)*gamma4*h/2*A*vv;
        vv=vv+(1-2*gamma2)*gamma4*h*B*uu;
        uu=uu+(1-gamma2)*gamma4*h/2*A*vv;
        vv=vv+gamma2*gamma4*h*B*uu;
        uu=uu+gamma2*(1-gamma4)*h/2*A*vv;
        vv=vv+gamma2*(1-2*gamma4)*h*B*uu;
        uu=uu+(1-gamma2)*(1-2*gamma4)*h/2*A*vv;
        vv=vv+(1-2*gamma2)*(1-2*gamma4)*h*B*uu;
        uu=uu+(1-gamma2)*(1-2*gamma4)*h/2*A*vv;
        vv=vv+gamma2*(1-2*gamma4)*h*B*uu;
        uu=uu+gamma2*(1-gamma4)*h/2*A*vv;
        vv=vv+gamma2*gamma4*h*B*uu;
        uu=uu+(1-gamma2)*gamma4*h/2*A*vv;
        vv=vv+(1-2*gamma2)*gamma4*h*B*uu;
        uu=uu+(1-gamma2)*gamma4*h/2*A*vv;
        vv=vv+gamma2*gamma4*h*B*uu;
        uu=uu+gamma2*gamma4*h*A*vv;
    end

    u(:,i+1)=uu-(gamma2*gamma4*h/2*A*vv);    v(:,i+1)=vv;
    E(i+1)=v(:,i+1)'*A*v(:,i+1)/2-u(:,i+1)'*B*u(:,i+1)/2;
end

%% output
figure
subplot(2,1,1);
plot(TimeSpan,E-E(1));
title('Energy fluctuation');
ylabel('Time');
subplot(2,1,2);
ss=sin(sqrt(-A*B)*TimeSpan);        % NOTE: this needs to be replaced when dim>1
cc=cos(sqrt(-A*B)*TimeSpan);        % NOTE: this needs to be replaced when dim>1
u_exact=cc*u0+sqrt(-A/B)*ss*v0;     
v_exact=-sqrt(-B/A)*ss*u0+cc*v0;    
plot(TimeSpan,sqrt(sum(([u-u_exact; v-v_exact]).^2,1)));
title('error in 2-norm (note the linear growth!)');
ylabel('Time');
