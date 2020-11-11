import numpy as np
import random
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF

def Monte_Carlo(ppc, varance):
    bus = ppc['bus']
    windF = ppc['windF']
    PV = ppc['PV']
    # 电负荷、气负荷、热负荷抽样，假定电负荷均服从正态分布
    load_e_p = np.random.normal(loc=abs(varance*bus[:,PD]), size=(bus.shape[0],1))
    load_e_q = p.random.normal(loc=abs(varance*bus[:,QD]), size=(bus.shape[0],1))

    # 风速威布尔分布  %假设风电只有 有功
    Pwind = np.zeros((windF.shape[0],1))
    Vwind = np.zeros((windF.shape[0],1))
    W_pro = 0   # 风机故障率
    W_break = np.ones((windF.shape[0],1))
    random = unifrnd(0,1,size(windF,1),1);  %生成连续均匀分布的随机数
    W_break(find(random-W_pro<0),1)=0;

    %判断风机是否被抽掉  风机出力
    for tt=1:size(windF,1)
        if windF(tt,PSET)==0
            Pwind(tt)=0;
        else
           if W_break(tt)==0
               windF(tt,PSET)=0;
               Pwind(tt)=0;
           else
                  Vwind(tt)=wblrnd(windF(tt,WF_C),windF(tt,WF_K),1,1);  %风速服从威布尔分布
                  if  Vwind(tt)> windF(tt,WF_VI) && Vwind(tt)<=windF(tt,WF_VRATE)
                      Pwind(tt)=windF(tt,PSET)*(Vwind(tt)-windF(tt,WF_VI))/(windF(tt,WF_VRATE)-windF(tt,WF_VI));%根据风速算功率
                  else
                      if Vwind(tt)> windF(tt,WF_VRATE) && Vwind(tt)<=windF(tt,WF_VOUT)
                          Pwind(tt)=windF(tt,PSET);
                      else
                          Pwind(tt)=0;
                      end
                  end
           end
        end
    end
    Ppv=zeros(size(PV,1),1);
    PV_pro=0;   %光伏电站故障率
    PV_break=ones(size(PV,1),1);
    random = unifrnd(0,1,size(PV,1),1);
    PV_break(find(random-PV_pro<0),1)=0;

    %光伏发电 出力
     for ii=1:size(PV,1)
        if PV(ii,PV_PMAX)==0
            Ppv(ii)=0;
        else
           if PV_break(ii)==0
               PV(ii,PV_PMAX)=0;
               Ppv(ii)=0;
           else
               Ppv(ii)=PV(ii,PV_PMAX)*betarnd(PV(ii,PV_ALPHA),PV(ii,PV_BETA));   %光伏功率计算
           end
        end
     end
    %%
    PQ_after=zeros(size(bus,1),1);  %新能源的功率

    for ii=1:size(windF,1)
        PQ_after(windF(ii,WF_BUS),1)=Pwind(ii)+ PQ_after(windF(ii,WF_BUS),1);
    end
    for ii=1:size(PV,1)
        PQ_after(PV(ii,PV_BUS),1)=Ppv(ii)+ PQ_after(PV(ii,PV_BUS),1);
    end

    for ii=1:size(bus,1)
        bus(:,PD)=load_e_p(:);
        bus(:,QD)=load_e_q(:);
    end

    bus(:,PD)=bus(:,PD)-PQ_after(:);
    %%
    %设计返回矩阵back
     back.bus=bus;
     back.version= mpc.version;
     back.baseMVA=mpc.baseMVA;
     back.gen=mpc.gen;
     back.branch=mpc.branch;
     back.gencost=mpc.gencost;
     renewable_record=[Pwind;Ppv];
     end


