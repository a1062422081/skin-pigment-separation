% skin pigment separation:
%  Sub-block selection algorithm

tic    %运行开始时间
clc;
clear;
[mpath,mname]=fileparts(mfilename('fullpath'));
% 1 读入图像数据
path='Data\2.jpg';
Cbase=imread(path);
Cpatent=Cbase(600:2900,100:1600,:);

iminfo=[800 2600 400 1400];
Cbase=Cbase(iminfo(1):iminfo(2),iminfo(3):iminfo(4),:);
        imshow(Cbase);
        title('原图');
        text(-500,500,{mname;num2str(iminfo)},'Interpreter','none');

        Cbr=double(Cpatent(:,:,1));   %将数据转成double类型
        Cbg=double(Cpatent(:,:,2));
        Cbb=double(Cpatent(:,:,3));
        row=size(Cbr,1);        %图像大小
        col=size(Cbr,2);
        
        %为了接下色素值可以进行计算和表达
        %对数据做适当放缩
        Cbr0_1=(Cbr(:)+1)/256;              
        Cbg0_1=(Cbg(:)+1)/256;
        Cbb0_1=(Cbb(:)+1)/256;
        
        Cblog=[log(Cbr0_1) log(Cbg0_1) log(Cbb0_1)]';
        Cbb_r=Cblog(2,:)-Cblog(1,:);
        Cbg_r=Cblog(3,:)-Cblog(1,:);
        Cb2d=[Cbb_r;Cbg_r];
        Cbr_r=Cblog(1,:);
        

num=0;
seinfo=[300 599 100 699];
figure
imshow(Cbase(seinfo(1):seinfo(2),seinfo(3):seinfo(4),:));

for n=[seinfo(1):5:seinfo(2)]
    for m=[seinfo(3):5:seinfo(4)]
        num=num+1;
        position(num,:)=[n m];
    end
end
       
Distance=[];
% r4=randperm(num,4)
r4=[1379]
for i=1:num
    n=position(i,1);
    m=position(i,2);
        p=Cbase(n:n+4,m:m+4,:);
        if ismember(i,r4)
            p1=p
            p2=roundn(log((double(p1)+1)/256),-3)

        end
        pr=double(p(:,:,1));
        pg=double(p(:,:,2));
        pb=double(p(:,:,3));
        pr=log((pr+1)/256);
        pg=log((pg+1)/256);
        pb=log((pb+1)/256);
        dB_R=pb-pr;
        dG_R=pg-pr;
        dBG_R=sum(sum(abs(dB_R)+abs(dG_R)))/25;
        Distance(i,:)=dBG_R;
end
        [D index]=sort(Distance,'descend');

        maxd=index(1:7200);
        maxp=position(maxd,:);
        p1=[];
        for i=1:1500     
            n=maxp(i,1);
            m=maxp(i,2);
            p1=[p1 Cbase(n:n+4,m:m+4,:)];
        end
            
        
        [V1 f1]=ICD_FastICA(p1);

toc    %运行结束时间
disp(['运行时间:',num2str(toc)]);

if f1==1
    V=abs(V1)
    %添加色素向量判断
    v1=V(:,1);
    v2=V(:,2);
    r1=V(2,1)/V(1,1);
    r2=V(2,2)/V(1,2);
    if r1>1&r1<=6.33
        V=[v2 v1];
    else if r2>=0.48&r2<1
            V=[v2 v1];
        end
    end
        
    D=(V^-1)*(Cb2d);
    
    
    C1Log=V*[D(1,:);zeros(1,size(D,2))];%+E;
    C1=[C1Log(1,:)+Cbr_r;C1Log(2,:)+Cbr_r];
    C1=exp(C1);
    
    C11=reshape(C1(1,:),row,col);
    C12=reshape(C1(2,:),row,col);
    
    figure;
    imshow(cat(3,0.87*(Cbr+1)/256,0.85*C11,0.85*C12));

    title('血色素');
    text(-500,500,{mname;num2str(iminfo);num2str(seinfo)},'Interpreter','none');
    
    C2Log=V*[zeros(1,size(D,2));D(2,:)];%+E;
    C2=[C2Log(1,:)+Cbr_r;C2Log(2,:)+Cbr_r];
    C2=exp(C2);
    
    C21=reshape(C2(1,:),row,col);
    C22=reshape(C2(2,:),row,col);
    
    figure;
        imshow(cat(3,0.8*(Cbr+1)/256,1.1*C21,1.1*C22));

    title('黑色素');
    text(-500,500,{mname;num2str(iminfo);num2str(seinfo)},'Interpreter','none');
end

%##################封装函数####################
function [V flag]=ICD_FastICA(C)
 %可手动设置区域大小
        Cr=double(C(:,:,1));   %将数据转成double类型
        Cg=double(C(:,:,2));
        Cb=double(C(:,:,3));
        
        %为了接下色素值可以进行计算和表达
        %对数据做适当放缩
        Cr0_1=(Cr(:)+1)/256;              
        Cg0_1=(Cg(:)+1)/256;
        Cb0_1=(Cb(:)+1)/256;
           
        Clog=[log(Cr0_1) log(Cg0_1) log(Cb0_1)]'; %转置后行为特征维度，即颜色分量，列为采样点数
        
        %-----------------------------下面是改动：通过作差降维，不使用PCA算法
        %通道作差
        Cb_r=Clog(2,:)-Clog(1,:);
        Cg_r=Clog(3,:)-Clog(1,:);
        C2d=[Cb_r;Cg_r];
        
        %2 数据预处理：去均值，PCA降维，白化
        
        X=C2d;        
        
        %-----------去均值---------
        
        [M,T] = size(X); %获取输入矩阵的行/列数，行数为观测数据的数目，列数为采样点数
        
        average=mean(X')';%均值
        
        for i=1:M                                
            
            X(i,:)=X(i,:)-average(i)*ones(1,T);
            
        end
        
        %---------白化/球化------
        
        Cx =cov(X',1); %计算协方差矩阵Cx
        
        [eigvector,eigvalue]= eig(Cx) %计算Cx的特征值和特征向量
        
        FV=eigvalue^(-1/2)*eigvector';%白化矩阵
        
        Z=FV*X;%正交矩阵
        
        
        %ICA分离
        
        %----------迭代-------
        
        Maxcount=10000;%最大迭代次数
        
        Critical=0.00001;%判断是否收敛
        
        m=M;%需要估计的分量的个数
        
        W=rand(m);
        
        %######设置分离矩阵的合格条件###
        flag=1;
        %######################
        
        for n=1:m
            
            if flag==1
                
                WP=W(:,n);%初始权矢量（任意）              
                
                count=0;
                
                LastWP=zeros(m,1);
                
                W(:,n)=W(:,n)/norm(W(:,n));
                
                while abs(WP-LastWP)>Critical & abs(WP+LastWP)>Critical    %当WP的方向不发生变化时，收敛
                    
                    count=count+1; %迭代次数
                    
                    LastWP=WP;%上次迭代的值
                    
                    
                    for i=1:m
                        
                        WP(i)=mean(Z(i,:).*(tanh((LastWP)'*Z)))-(mean(1-(tanh((LastWP))'*Z).^2)).*LastWP(i);
                        
                    end
                    
                    WPP=zeros(m,1);
                    
                    for j=1:n-1
                        
                        WPP=WPP+(WP'*W(:,j))*W(:,j);
                        
                    end
                    
                    WP=WP-WPP;
                    
                    WP=WP/(norm(WP));
                    
                    if count==Maxcount
                        
                        fprintf('未找到相应的信号');
                        flag=0;
                        
                        break;
                        % return;
                        
                    end
                    
                end
                
                W(:,n)=WP;
                
            end
            
            
        end
                %---------------迭代结束---------------
        if flag==1
            W
            
            V=inv(FV)*(W')^-1;
            
            %矩阵矩阵列向量标准化
            
            vnorm=sqrt(sum(V.^2));
            V=V./repmat(vnorm,size(V,1),1);
           
        else
            V=zeros(2,2);
        end
end
















