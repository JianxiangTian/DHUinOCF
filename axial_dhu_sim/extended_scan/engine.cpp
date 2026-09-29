// Overdamped soft-repulsive particles, explicit reservoirs, distant GC baths.
// Units: um, s, kBT. No periodic recycling in open geometry.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
const double PI=acos(-1.0);
struct P {double z,x,fz=0,fx=0;};
double fold(double x,double L){x=fmod(x,2*L);if(x<0)x+=2*L;return x>L?2*L-x:x;}
double wrap(double x,double L){x=fmod(x,L);return x<0?x+L:x;}
double pairU(double r,double eps){return r<1?.5*eps*(1-r)*(1-r):0.;}
struct Sim {
 double L=256,R=48,W=16,D=.2,u=.2,dt=.02,eps=20,kappa=0,screen=0,activity=.125;
 double burn=2400,duration=6000,save_dt=5,control_dt=.2,bath_rate=40,bath_width=8,lambda_ref=2;
 bool periodic=false;int seed=31,Nfix=512;string prefix;double T,lo,hi;
 vector<P> p;vector<int> head,next;int nz,nx;vector<double> force,rho,smooth;
 mt19937_64 gen;normal_distribution<double> gauss{0,1};uniform_real_distribution<double> uniform{0,1};
 long long inserted=0,removed=0,attemptI=0,attemptD=0,fluxL=0,fluxR=0;
 long long insL=0,delL=0,insR=0,delR=0;int maxbalance=0,Ninitial,Cinitial;
 double max_pair_force=0,closest=1e9;long long closepairs=0,deep_pairs=0,measures=0;
 double zsep(double a,double b){double dz=a-b;if(periodic)dz-=T*round(dz/T);return dz;}
 int countControl(){int n=0;for(auto &v:p)n+=(v.z>=lo&&v.z<hi);return n;}
 void cells(){
   fill(head.begin(),head.end(),-1);next.resize(p.size());
   for(int i=0;i<(int)p.size();++i){int iz=min(nz-1,(int)p[i].z),ix=min(nx-1,(int)p[i].x);int c=iz*nx+ix;next[i]=head[c];head[c]=i;}
 }
 void repulsion(){
   for(auto &a:p)a.fz=a.fx=0;
   if(eps==0)return;
   cells();
   for(int i=0;i<(int)p.size();++i){
     int iz=min(nz-1,(int)p[i].z),ix=min(nx-1,(int)p[i].x);
     for(int dzc=-1;dzc<=1;++dzc){int zc=iz+dzc;if(periodic)zc=(zc+nz)%nz;else if(zc<0||zc>=nz)continue;
       for(int xc=max(0,ix-1);xc<=min(nx-1,ix+1);++xc)
         for(int j=head[zc*nx+xc];j!=-1;j=next[j])if(j>i){
           double dz=zsep(p[i].z,p[j].z),dx=p[i].x-p[j].x,r2=dz*dz+dx*dx;
           if(r2<1&&r2>1e-24){double r=sqrt(r2),f=eps*(1-r)/r;
             p[i].fz+=f*dz;p[j].fz-=f*dz;p[i].fx+=f*dx;p[j].fx-=f*dx;
             max_pair_force=max(max_pair_force,eps*(1-r));
           }
         }
     }
   }
 }
 double energy(double z,double x,int skip=-1){
   if(eps==0)return 0;
   double en=0;for(int i=0;i<(int)p.size();++i)if(i!=skip){
     double dz=zsep(z,p[i].z);if(abs(dz)>=1)continue;double dx=x-p[i].x;
     if(abs(dx)>=1)continue;double r2=dz*dz+dx*dx;if(r2<1)en+=pairU(sqrt(r2),eps);
   }return en;
 }
 void bath(bool right){
   double a=right?T-bath_width:0,b=a+bath_width;
   // Continuous-time grand-canonical birth/death generator. Insertion proposals
   // have rate gamma*zeta*V, accept exp(-Delta U); each bath particle dies at
   // rate gamma. Both rates obey local detailed balance. With eps=0 this is a
   // linear immigration/death process, avoiding number-dependent admission noise.
   double zetaV=activity*bath_width*W,gamma=bath_rate/zetaV,clock=0;
   while(true){
     vector<int> eligible;for(int i=0;i<(int)p.size();++i)if(p[i].z>=a&&p[i].z<b)eligible.push_back(i);
     int n=eligible.size();double rate=bath_rate+gamma*n;
     clock+=-log(max(1e-300,uniform(gen)))/rate;if(clock>dt)break;
     if(uniform(gen)<bath_rate/rate){
       ++attemptI;double z=a+bath_width*uniform(gen),x=W*uniform(gen);
       double accept=exp(-energy(z,x));
       if(uniform(gen)<accept){p.push_back({z,x});++inserted;if(right)++insR;else ++insL;}
     }else{
       ++attemptD;if(n==0)continue;int i=eligible[min(n-1,(int)(uniform(gen)*n))];
       p[i]=p.back();p.pop_back();++removed;if(right)++delR;else ++delL;
     }
   }
 }
 void feedback(){
   int G=rho.size();fill(rho.begin(),rho.end(),0);
   for(auto &v:p)if(v.z>=lo&&v.z<hi){int j=min(G-1,(int)(v.z-lo));rho[j]+=1.;}
   double mean=0;for(auto v:rho)mean+=v/G;for(auto &v:rho)v-=mean;
   // Eight reflected/p eriodic nearest-neighbor smoothing passes, variance 4 um^2.
   for(int pass=0;pass<8;++pass){for(int j=0;j<G;++j){
     int l=j?j-1:(periodic?G-1:0),r=j<G-1?j+1:(periodic?0:G-1);
     smooth[j]=.25*rho[l]+.5*rho[j]+.25*rho[r];}rho.swap(smooth);}
   double K=kappa/(D*lambda_ref);
   if(screen==0){
     force[0]=0;for(int j=0;j<G;++j)force[j+1]=force[j]+K*rho[j];
     if(periodic){double avg=0;for(int j=0;j<G;++j)avg+=force[j]/G;for(auto &v:force)v-=avg;force[G]=force[0];}
     else force[G]=0;
   }else if(!periodic){
     vector<double> cp(G),dp(G),phi(G);double s2=screen*screen;
     for(int j=0;j<G;++j){double diag=((j==0||j==G-1)?1.:2.)+s2;
       double den=diag+(j?cp[j-1]:0);cp[j]=j==G-1?0:-1./den;dp[j]=(K*rho[j]+(j?dp[j-1]:0))/den;}
     phi[G-1]=dp[G-1];for(int j=G-2;j>=0;--j)phi[j]=dp[j]-cp[j]*phi[j+1];
     force[0]=force[G]=0;for(int j=1;j<G;++j)force[j]=phi[j-1]-phi[j];
   }else{
     // Small O(G^2) cosine/sine solve used only at controller updates.
     fill(force.begin(),force.end(),0);
     for(int m=1;m<=G/2;++m){double q=2*PI*m/G,rr=0,ii=0,den=4*pow(sin(q/2),2)+screen*screen;
       for(int j=0;j<G;++j){rr+=rho[j]*cos(q*(j+.5));ii-=rho[j]*sin(q*(j+.5));}
       double scale=(m==G/2?1.:2.)*K/G/den;
       for(int j=0;j<G;++j)force[j]+=scale*2*sin(q/2)*(rr*sin(q*j)+ii*cos(q*j));
     }force[G]=force[0];
   }
 }
 double fb(double z){if(kappa==0||z<lo||z>=hi)return 0;double t=z-lo;int j=min((int)L-1,(int)t);double f=t-j;return force[j]*(1-f)+force[j+1]*f;}
 void step(){
   int oldc=countControl();long long fL=fluxL,fR=fluxR;repulsion();double noise=sqrt(2*D*dt);
   for(auto &a:p){double oldz=a.z;double z=a.z+(u+D*(a.fz+fb(a.z)))*dt+noise*gauss(gen);
     a.x=fold(a.x+D*a.fx*dt+noise*gauss(gen),W);a.z=periodic?wrap(z,T):fold(z,T);
     if(!periodic){fluxL+=(oldz<lo&&a.z>=lo)-(oldz>=lo&&a.z<lo);fluxR+=(oldz<hi&&a.z>=hi)-(oldz>=hi&&a.z<hi);}
   }
   if(!periodic){bath(false);bath(true);int delta=countControl()-oldc;maxbalance=max(maxbalance,(int)abs(delta-((fluxL-fL)-(fluxR-fR))));}
 }
 void sample(double t,ofstream &out){
   vector<double> row;int nc=countControl();double nl=0,nr=0;
   for(auto &a:p){nl+=a.z<lo;nr+=a.z>=hi;}
   row={t,(double)p.size(),(double)nc,nl,nr,(double)fluxL,(double)fluxR,(double)inserted,(double)removed,
       (double)insL,(double)delL,(double)insR,(double)delR};
   // Density profile over entire simulated domain, 48 fixed bins.
   vector<double> profile(48,0);for(auto &a:p)++profile[min(47,(int)(48*a.z/T))];row.insert(row.end(),profile.begin(),profile.end());
   double O=L/2;double starts[3]={lo+L/16,lo+L/4,lo+7*L/16};
   for(double start:starts){vector<double> loc;for(auto &a:p)if(a.z>=start&&a.z<start+O)loc.push_back(a.z-start);
     row.push_back(loc.size());for(int m=1;m<=16;++m){double re=0,im=0;for(double z:loc){re+=cos(2*PI*m*z/O);im-=sin(2*PI*m*z/O);}row.push_back(re);row.push_back(im);}
   }
   double start=lo+L/4;double lengths[5]={4,8,16,32,L/4};
   for(double len:lengths)for(int j=0;j<24;++j){double a=start+(O-len)*(j+.5)/24;int n=0;for(auto &p0:p)n+=p0.z>=a&&p0.z<a+len;row.push_back(n);}
   // Counts of close/deeply overlapping pairs inside central observation region.
   double close=0,deep=0,minr=1e9;
   for(int i=0;i<(int)p.size();++i)if(p[i].z>=start&&p[i].z<start+O)
     for(int j=i+1;j<(int)p.size();++j)if(p[j].z>=start&&p[j].z<start+O){double dz=zsep(p[i].z,p[j].z);if(abs(dz)>=1)continue;double dx=p[i].x-p[j].x,r2=dz*dz+dx*dx;if(r2<1){++close;minr=min(minr,sqrt(r2));if(r2<.25)++deep;}}
   row.push_back(close);row.push_back(deep);row.push_back(minr==1e9?1.:minr);

   // Size-scaling observables. No random numbers: unchanged physical trajectory.
   // Columns 283:290: Ncontrol, full-control complex m=1,2, Neumann cos m=1,2.
   row.push_back(nc);
   for(int m=1;m<=2;++m){double re=0,im=0;for(auto &v:p)if(v.z>=lo&&v.z<hi){double phase=2*PI*m*(v.z-lo)/L;re+=cos(phase);im-=sin(phase);}row.push_back(re);row.push_back(im);}
   for(int m=1;m<=2;++m){double re=0;for(auto &v:p)if(v.z>=lo&&v.z<hi)re+=cos(PI*m*(v.z-lo)/L);row.push_back(re);}
   // Each probe: N, rectangular complex m=1, Hann complex m=2, sum w^2,
   // rectangular complex m=2. Hann m=2 is orthogonal to a uniform density mode.
   // Path A: O=L/2. Path B: O=sqrt(32 um * L), so O/L -> 0 as L -> infinity.
   double probe_lengths[2]={L/2,sqrt(32*L)};
   for(double len:probe_lengths){double start=lo+(L-len)/2;
     double n=0,re=0,im=0,rt=0,it=0,w2=0,re2=0,im2=0;
     for(auto &v:p)if(v.z>=start&&v.z<start+len){double a=(v.z-start)/len,theta=2*PI*a,w=pow(sin(PI*a),2);
       ++n;re+=cos(theta);im-=sin(theta);rt+=w*cos(2*theta);it-=w*sin(2*theta);w2+=w*w;re2+=cos(2*theta);im2-=sin(2*theta);}
     for(double v:{n,re,im,rt,it,w2,re2,im2})row.push_back(v);
   }
   // Columns 306:426: five count lengths within mesoscopic centered probe,
   // each evaluated at 24 fixed locations. Last length = O_meso/2.
   double Om=sqrt(32*L),zm=lo+(L-Om)/2;
   for(double frac:{.125,.25,.5,.75,1.}){double len=frac*Om/2;
     for(int j=0;j<24;++j){double a=zm+(Om-len)*(j+.5)/24;int n=0;for(auto &v:p)n+=v.z>=a&&v.z<a+len;row.push_back(n);}
   }


   // Deterministic extra observations, with no change to forces or random draws.
   int M=max(64,(int)ceil(.15*L/(4*PI))+2);
   vector<double> rr(M+1,0),ii(M+1,0);
   double z0=lo+L/4, len=L/2;
   for(auto &v:p)if(v.z>=z0&&v.z<z0+len){
     double theta=2*PI*(v.z-z0)/len,ct=cos(theta),st=sin(theta),c=1,s=0;
     for(int m=0;m<=M;++m){rr[m]+=c;ii[m]-=s;double cn=c*ct-s*st;s=s*ct+c*st;c=cn;}
   }
   for(int m=0;m<=M;++m){row.push_back(rr[m]);row.push_back(ii[m]);}

   out.write((char*)row.data(),row.size()*sizeof(double));++measures;
 }
 void init(){
   T=periodic?L:L+2*R;lo=periodic?0:R;hi=lo+L;gen.seed(seed);
   nz=(int)ceil(T);nx=(int)ceil(W);head.resize(nz*nx);rho.resize((int)L);smooth.resize((int)L);force.resize((int)L+1);
   int n=periodic?Nfix:poisson_distribution<int>(activity*W*T)(gen);p.reserve(2*n);
   // Avoid pathological initial overlap without imposing an axial lattice.
   for(int i=0;i<n;++i){double z=0,x=0;int tries=0;do{z=T*uniform(gen);x=W*uniform(gen);++tries;}while(eps>0&&energy(z,x)>1&&tries<10000);p.push_back({z,x});}
   Ninitial=p.size();Cinitial=countControl();feedback();
 }
 void run(){
   init();ofstream data(prefix+".bin",ios::binary);long steps=llround((burn+duration)/dt),save=llround(save_dt/dt),ctrl=llround(control_dt/dt);
   for(long n=0;n<steps;++n){if(n%ctrl==0)feedback();step();if((n+1)*dt>burn&&(n+1)%save==0)sample((n+1)*dt,data);}
   data.close();ofstream snapshot(prefix+"_snapshot.csv");snapshot<<"z,x\n";for(auto &v:p)snapshot<<v.z<<","<<v.x<<"\n";
   ofstream meta(prefix+"_meta.json");meta<<setprecision(14)<<"{\n";
   meta<<"\"seed\":"<<seed<<",\"periodic\":"<<(periodic?"true":"false")<<",\"L\":"<<L<<",\"R\":"<<R<<",\"W\":"<<W<<",\"T\":"<<T<<",\"D\":"<<D<<",\"u\":"<<u<<",\"dt\":"<<dt<<",\"eps\":"<<eps<<",\"kappa\":"<<kappa<<",\"screen\":"<<screen<<",\"activity\":"<<activity<<",\"lambda_ref\":"<<lambda_ref<<",\"burn\":"<<burn<<",\"duration\":"<<duration<<",\"save_dt\":"<<save_dt<<",\"control_dt\":"<<control_dt<<",\"bath_rate\":"<<bath_rate<<",\"bath_width\":"<<bath_width<<",";
   meta<<"\"columns\":"<<(426+2*(max(64,(int)ceil(.15*L/(4*PI))+2)+1))<<",\"frames\":"<<measures<<",\"N_initial\":"<<Ninitial<<",\"N_final\":"<<p.size()<<",\"C_initial\":"<<Cinitial<<",\"C_final\":"<<countControl()<<",\"inserted\":"<<inserted<<",\"removed\":"<<removed<<",\"flux_left\":"<<fluxL<<",\"flux_right\":"<<fluxR<<",\"max_control_balance_error\":"<<maxbalance<<",\"total_balance_error\":"<<((long long)p.size()-Ninitial-inserted+removed)<<",\"max_pair_force\":"<<max_pair_force<<",\"attempt_insert\":"<<attemptI<<",\"attempt_delete\":"<<attemptD<<"\n}";
   cout<<prefix<<" done N="<<p.size()<<" balance="<<maxbalance<<endl;
 }
};
int main(int argc,char**argv){
 map<string,string>a;for(int i=1;i+1<argc;i+=2)a[argv[i]]=argv[i+1];Sim s;
 if(a.count("--selftest")){
   double r=.73,h=1e-6,e=20,fd=-(pairU(r+h,e)-pairU(r-h,e))/(2*h),exact=e*(1-r);
   if(abs(fd-exact)>1e-7)return 2;
   s.L=32;s.R=16;s.W=8;s.eps=20;s.init();s.repulsion();double fx=0,fz=0;for(auto &p:s.p){fx+=p.fx;fz+=p.fz;}if(abs(fx)+abs(fz)>1e-8)return 3;
   cout<<"PASS force gradient and pair antisymmetry"<<endl;return 0;
 }
 auto get=[&](string k,double &v){if(a.count(k))v=stod(a[k]);};
 get("--L",s.L);get("--R",s.R);get("--W",s.W);get("--u",s.u);get("--dt",s.dt);get("--eps",s.eps);get("--kappa",s.kappa);get("--screen",s.screen);get("--activity",s.activity);get("--burn",s.burn);get("--duration",s.duration);get("--bath-rate",s.bath_rate);get("--control-dt",s.control_dt);get("--lambda-ref",s.lambda_ref);
 if(a.count("--seed"))s.seed=stoi(a["--seed"]);if(a.count("--periodic"))s.periodic=stoi(a["--periodic"]);if(a.count("--N"))s.Nfix=stoi(a["--N"]);
 s.prefix=a.count("--out")?a["--out"]:"run";
 if(s.L<64||s.L!=floor(s.L)||s.R<16||s.W!=floor(s.W)||s.dt<=0||s.bath_width>=s.R||s.kappa<0){cerr<<"invalid parameters";return 4;}
 s.run();
}
