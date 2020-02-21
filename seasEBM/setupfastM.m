function [invM,Mh]=setupfastM(delx,jmx,D,B,Cl,delt);
%delx,jmx,D,0.,1.0,delt

%set up lambda array.
lam=(1-[-1:delx:1]'.^2)/delx^2;
lam=D(:).*lam(:);
disp(num2str(lam(1)))

M = zeros(jmx,jmx);
M(1,1) = -B - lam(2);
M(1,2) = lam(2);

M(jmx,jmx-1) = lam(jmx);
M(jmx,jmx)   = -B - lam(jmx);

for j=2:jmx-1
  M(j,j-1) = lam(j);
  M(j,j)   = -B - (lam(j+1)+lam(j));
  M(j,j+1) = lam(j+1);
end

%add in heat capacities
M=M/Cl;

%calculate the inverse of M', the matrix operator.
 Mh=M;
 M=0.5*M;
 for j=1:jmx
   M(j,j)=M(j,j) - 1./delt;
 end
 invM = inv(M);

