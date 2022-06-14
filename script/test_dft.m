%% 测试频域滤波的效果
clc, clear, close all;
% A = imread('../resource/xx-grating_002_0.bmp');
A = imread('../resource/noised_0.5.bmp');
% x = A;
x = rgb2gray(A);

% y=imnoise(x,'gaussian',0,0.02);  %加入均值为0，方差为0.02的高斯噪声
y = x;
f = double(y); %将数据类型转换成double
F = fft2(f); %二维傅里叶变换
F1 = fftshift(F); %中心化，将低频频谱由四周转换到中心，距离中心越远的频率越高
[M, N] = size(F);
% n = 2;  %巴特沃斯低通滤波器的阶数
D0 = 50;  %设置截止频率为50
m = fix(M/2);
n = fix(N/2);
for i=1:M  
    for j=1:N
        D = sqrt((i-m)^2+(j-n)^2);
        % h1 = 1 / (1 + (D / D0)^(2 * n));  %计算巴特沃兹低通滤波器
        h2 = exp(-(D.^2)./(2 * (D0^2)));  %计算高斯低通滤波器
        % G1(i,j) = h1 * F1(i,j);  
        G2(i,j) = h2 * F1(i,j);
    end
end
% G1 = ifftshift(G1);  %将巴特沃兹低通滤波后的频谱反中心化
% G11 = uint8(real(ifft2(G1)));  %取二维傅里叶反变换后的实部
G2 = ifftshift(G2);  %将高斯低通滤波后的频谱反中心化
G22 = uint8(real(ifft2(G2)));  %取二维傅里叶反变换后的实部
% subplot(2,4,1),imshow(x),title('原图');
% subplot(2,4,2),imshow(y),title('加入高斯噪声后');
% subplot(2,4,3),imshow(G11),title('二阶巴特沃斯低通滤波后图像');
% subplot(2,4,4),imshow(G22),title('高斯低通滤波后图像');
% subplot(2,4,5),imshow(log(1+abs(F)),[ ]),title('加噪图像的频谱');
% subplot(2,4,6),imshow(log(1+abs(F1)),[ ]),title('频谱中心化');
% subplot(2,4,7),imshow(log(1+abs(G1)),[ ]),title('巴特沃兹低通滤波器频谱反中心化');
% subplot(2,4,8),imshow(log(1+abs(G2)),[ ]),title('高斯低通滤波器频谱反中心化');


subplot(2,2,1),imshow(x),title('原图');
subplot(2,2,2),imshow(G22),title('高斯低通滤波后图像');
subplot(2,2,3),imshow(log(1+abs(F)),[ ]),title('加噪图像的频谱');
subplot(2,2,4),imshow(log(1+abs(F1)),[ ]),title('频谱中心化');
figure; imshow(log(1+abs(G2)),[ ]),title('高斯低通滤波器频谱反中心化');