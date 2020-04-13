
clearvars;
close all;
clc;

shouldPlot = 1;

%filename = 'Harray2.8GhzULA8x8s003.mat'
filename = 'Harray60GhzULA8x8s006.mat'
load(filename)

% one receiver
Ht = Ht(:,:,1,:,:);

%permute dimensions before reshape: scenes before episodes
Ht = permute(Ht, [2,1,3,4,5]);

s = size(Ht);
reshape_dim = prod(s(1:3));

Harray = reshape(Ht, reshape_dim, s(4),s(5));
Hvirtual = zeros(size(Harray));
scalingFactor = 1 / sqrt(s(4) * s(5));
for i = 1:1:reshape_dim
    m = squeeze(Harray(i,:,:));
    Hvirtual(i, : ,:) = scalingFactor*fft2(m);
    if shouldPlot
        imagesc(abs(squeeze(Hvirtual(i,:,:))));
        episode = floor((i-1)/10)+1;
        scene = mod(i-1,10)+1;
        title(sprintf('episode = %03d, scene = %02d\n', episode, scene))
        drawnow
    end
end

save('channel_data.mat', 'Harray', 'Hvirtual');

# split train and test data
# expand to 10 users 
