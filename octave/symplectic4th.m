%% 4th-order symplectic integrator for u'=Av, v'=Bu (H(u,v)=v'Av/2-u'Bu/2)
clear;
clc;

%% parameters
A=1;
B=-1;
dim=size(A,2);
u0=1;    v0=0.3;        % NOTE: this can be changed
T=100;
h=0.01;
gamma=1/(2-2^(1/3));

%% symplectic integration
TimeSpan=[0:h:T];   TimeSteps=length(TimeSpan)-1;
u=zeros(dim,TimeSteps+1);   u(:,1)=u0;
v=zeros(dim,TimeSteps+1);   v(:,1)=v0;
E=zeros(1,TimeSteps+1);
uu=u(:,1);  vv=v(:,1);  E(1)=vv'*A*vv/2-uu'*B*uu/2;
for i=1:TimeSteps
    uu=uu+gamma*h/2*A*vv;
    vv=vv+gamma*h*B*uu;
    uu=uu+(1-gamma)*h/2*A*vv;
    vv=vv+(1-2*gamma)*h*B*uu;
    uu=uu+(1-gamma)*h/2*A*vv;
    vv=vv+gamma*h*B*uu;
    uu=uu+gamma*h/2*A*vv;
    
    u(:,i+1)=uu;    v(:,i+1)=vv;
    E(i+1)=vv'*A*vv/2-uu'*B*uu/2;
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
title('error in 2-norm');
ylabel('Time');
