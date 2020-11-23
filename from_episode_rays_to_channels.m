function [allChannels,numOfInvalidChannels] = ...
    from_episode_rays_to_channels(fileName,numTx,numRx,...
    normalizedSpacingTx,normalizedSpacingRx)
%function [allChannels,numOfInvalidChannels] = ...
%    from_episode_rays_to_channels(fileName,numTx,numRx,...
%    normalizedSpacingTx,normalizedSpacingRx)
%Convert rays of an episode into MIMO matrices using narrowband 
%(not the wideband) model.
%Aldebaro. Nov 21, 2018.

%read all ray data
allEpisodeData=readAllEpisodeData(fileName);
[numScenes, numTxRxPairs, maxNumPaths, numPathParameters] = ...
    size(allEpisodeData);

%pre-allocate with NaNs
allChannels = NaN*ones(numScenes, numTxRxPairs,numRx,numTx);
numOfInvalidChannels = 0;
for s=1:numScenes
    for r=1:numTxRxPairs
        channelRays=channelRaysDiscardingInvalids(allEpisodeData,s,r);
        if  channelRays == -1 %value returned for all channels invalid
            numOfInvalidChannels = numOfInvalidChannels + 1;            
            continue %next Tx / Rx pair
        end        
        [numPaths, numParameters] = size(channelRays);
        %Insite adopts theta as elevation and phi as azimuth
        %path summary:
        %<path number>
        %<total interactions for path> (not including Tx and Rx)
        %<received power(dBm)>
        %<time of arrival(sec)>
        %<arrival theta(deg)>  => elevation
        %<arrival phi(deg)> => azimuth
        %<departure theta(deg)> => elevation
        %<departure phi(deg)> => azimuth
        %InSite provides angles in degrees. Convert to radians
        %Note that Python wrote departure first, while InSite writes
        %arrival.
        channelMagnitude = channelRays(:,1); %gainMagnitude in dB;
        %timeOfArrival = channelRays(:,2); %timeOfArrival;
        %AoD_el = channelRays(:,3); %AoD_el;
        AoD_az = channelRays(:,4); %.AoD_az;
        %AoA_el = channelRays(:,5); %AoA_el;
        AoA_az = channelRays(:,6); %.AoA_az;
        %isLOS = channelRays(:,7);
        channelPhase = channelRays(:,8); %.rayPhases;
        channelMagnitude = 10.^(0.1*channelMagnitude); %dB to linear
        complexGains=channelMagnitude .* exp(1j * channelPhase);
        %isChannelLOS = sum(isLOS);
        H=narrowbandULAsMIMOChannel(numTx,numRx,normalizedSpacingTx,...
            normalizedSpacingRx,AoA_az, AoD_az, complexGains);    
        allChannels(s,r,:,:)=H;
    end
end
