%%
clc
clear

restoredefaultpath;

addpath(genpath('../build/lib/'))

%% set everything up
number_of_runs = 10;

%load data
I = im2double(imread('../data/intensity.png'));
lin_op = im2double(imread('../data/shading.png'));%only pixelwise
mask = logical(imread('../data/mask.png'));

%set parameters
lambda = 0.1;
alpha = -1; %piecewise constant segmentation
% alpha = 10; %piecewise smooth segmentation
max_iter = 5000;
tol = 1e-5;

verbose = 1;

%low-level parameters
gamma = 2;
tau = 0.25;
sigma = 0.5;


%% mumfordShah

% initiate c++/cuda
tic;
ms = mumfordShahMEX('initMumfordShah', lambda, alpha, max_iter, tol, gamma, verbose, tau, sigma);
t_init = toc;

% set the data
tic;
mumfordShahMEX('setDataMumfordShah', ms,	single(I), mask);
t_set = toc;

tic;
for ii=1:number_of_runs
  result = double(mumfordShahMEX('runMumfordShah', ms,  single(lin_op)));
end
t_run = toc;

tic;
mumfordShahMEX('closeMumfordShah', ms);
t_close = toc;

%% show
fprintf('Timings:\nt_init: %f\nt_set: %f\nt_run: %f\nt_close: %f\n', t_init,t_set,t_run/number_of_runs,t_close);

figure(1);
subplot(2,2,1);
imshow(I,[]);
title('input image');

figure(1);
subplot(2,2,2);
imshow(mask,[]);
title('mask');

figure(1);
subplot(2,2,3);
imshow(lin_op,[]);
title('pixel-wise linear operator');

figure(1);
subplot(2,2,4);
imshow(result,[]);
title('result');
