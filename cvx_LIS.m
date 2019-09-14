M = 8;
N = 64;
rou = 1;
Q_list = [];
total_num = 30000;
G_list = 1/sqrt(2)*(randn(total_num,M,N)+1j*randn(total_num,M,N));
hr_list = 1/sqrt(2)*(randn(total_num,N,1)+1j*randn(total_num,N,1));
hd_list = 1/sqrt(2)*(randn(total_num,M,1)+1j*randn(total_num,M,1));
tic
for i=1:1:total_num
    G = squeeze(G_list(i,:,:));
    hr = hr_list(i,:,:);
    hd = conj(hd_list(i,:,:)');
    Dh = diag(hr);
    v = diag((Dh')*(G')*G*Dh);
    V = rou*(1-rou)*diag(v);
    V(end+1,:)=0;
    V(:,end+1)=0;
    R1 = rou*rou*(Dh')*(G')*G*Dh;
    R2 = rou*(Dh')*(G')*hd;
    R3 = rou*(hd')*G*Dh;
    R = [R1,R2;R3,0];
    cvx_begin sdp quiet
        variable Q(N+1,N+1) hermitian toeplitz;
        minimize (-real(trace((R+V)*Q)));
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
save('./data/test_dataset_8_and_64','G_list','hr_list','hd_list','Q_list');
