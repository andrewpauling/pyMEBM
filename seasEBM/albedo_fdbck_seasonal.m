function alb=albedo_fdbck_seasonal(T,jmx,x);

% recalculate albedo.

alb=ones(jmx,1)*0.3;

%alternative albedo that depends on latitude (zenith angle)
%alb=0.31+0.08*(3*x.^2-1)/2;

 k=find(T<=-5);
 alb(k)=0.6;
