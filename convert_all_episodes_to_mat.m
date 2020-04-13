%Convert all hdf5 files into a single mat file.
%Aldebaro. Dec 4, 2019.
numEpisodes = 1000; %number of episodes to be processed

%compose as fileName = [insitePath filePrefix num2str(episode) extension]
insitePath='D:\ak\Students\Wesin2013\2019\china_ch_estimation_results\';
%insitePath='C:\Users\Brenda\Documents\codesLASSE\insitedata\';
addpath('../ray_tracing_io/');
addpath('../antenna_array/');
filePrefix='china_5s_e';
extension='.hdf5';
expected_numTxRxPairs = 10; %expected number of receivers (assuming 1 transmitter)
expected_numScenes = 1; %expected number of scenes per episode

numTx = 8;                   %TX ULA: number of antenna elements
numRx = 8;                   %RX ULA: number of antenna elements
freq = 60*10^9; %carrier frequency in Hz
normalizedSpacing = 0.5; %Normalized spacing: distance between 2 ULA

%pre-allocate space
Ht = NaN * ones(numEpisodes, expected_numScenes, expected_numTxRxPairs, numRx, numTx);

total_num_invalid_channels = 0;
for e=0:numEpisodes-1 %go over all episodes, assume first file is 0 (not 1)
    fileName = [insitePath filePrefix num2str(e) extension];
    disp(['Processing ' fileName])
    %read ray data
    [allChannels,numOfInvalidChannels] = ...
    from_episode_rays_to_channels(fileName,numTx,numRx,...
            normalizedSpacing,normalizedSpacing);
    total_num_invalid_channels = total_num_invalid_channels + numOfInvalidChannels;    
    [numScenes, numTxRxPairs,numRx,numTx] = size(allChannels);
    %total_num_of_TxRxPairs = total_num_of_TxRxPairs + numTxRxPairs;
    if numTxRxPairs ~= expected_numTxRxPairs
        warning('Logic error: numTxRxPairs ~= expected_numTxRxPairs')
    end
    if numScenes ~= expected_numScenes
        warning('Logic error: numScenes ~= expected_numScenes')
    end
    Ht(e+1,:,:,:,:) = allChannels; %add 1 because e starts at 0
end
disp(['Processed total numOfInvalidChannels = ' num2str(total_num_invalid_channels)])

%write output file
output_file_name = [filePrefix num2str(numEpisodes) '.mat'];
eval(['save -v6 ' output_file_name ' Ht'])
disp(['Wrote ' output_file_name]);

%imagesc(squeeze(abs(Ht(1,1,1,:,:)))) %to visualize a channel