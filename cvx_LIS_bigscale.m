M = 8;
N = 32;
rou = 1;
Q_list = [];
total_num = 10000;

%path_loss for channel between BS and LIS is fixed
d0 = 8*rand(total_num,1);
d1 = 1+5*rand(total_num,1);  
dBU = sqrt(d0.^2+d1.^2);
dLU = sqrt((8-d0).^2+d1.^2);
PL_BU =  -20.4*log10(dBU);
big_scale_BU_linear = sqrt(10.^(PL_BU/10));
PL_LU =  -20.4*log10(dLU);
big_scale_LU_linear = sqrt(10.^(PL_LU/10));
PL_BL = -20.4*log10(ones(total_num,1)*8);
big_scale_BL_linear = sqrt(10.^(PL_BL/10));

G_list = 1/sqrt(2)*(randn(total_num,M,N)+1j*randn(total_num,M,N));
big_scale_BL_linear = repmat(big_scale_BL_linear,[1,M,N]);
G_list = G_list.*big_scale_BL_linear;
hr_list = 1/sqrt(2)*(randn(total_num,N,1)+1j*randn(total_num,N,1));
big_scale_LU_linear = repmat(big_scale_LU_linear,[1,N,1]);
hr_list = hr_list.*big_scale_LU_linear;
hd_list = 1/sqrt(2)*(randn(total_num,M,1)+1j*randn(total_num,M,1));
big_scale_BU_linear = repmat(big_scale_BU_linear,[1,M,1]);
hd_list = hd_list.*big_scale_BU_linear;

tic
for i=1:1:total_num
    G = squeeze(G_list(i,:,:));
    hr = hr_list(i,:,:);
    hd = conj(hd_list(i,:,:)');
    Dh = diag(hr);
    R1 = rou*rou*(Dh')*(G')*G*Dh;
    R2 = rou*(Dh')*(G')*hd;
    R3 = rou*(hd')*G*Dh;
    R = [R1,R2;R3,0];
    cvx_begin sdp quiet
        variable Q(N+1,N+1) hermitian;
        minimize (-real(trace(R*Q)));
        subject to
            diag(Q)==1;
            Q == hermitian_semidefinite(N+1);
    cvx_end
    Q_list = [Q_list,Q];
    if mod(i,100)==0
        i
    end
end
toc
save('./data/bigscale/test_dataset_8_and_8','G_list','hr_list','hd_list','Q_list');
