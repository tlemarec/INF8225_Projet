clear
fileID = fopen('./log.txt','r');

% Read first 19 lines
for i = 1:18
    line = fgetl(fileID); 
end

% Get time and reward data from next lines
C = textscan(fileID,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s');
epoch = length(C{1});

% Get time
time_v = [];
for i = 1:epoch
    hour_v = str2double(C{5}{i}(1:2));
    min_v =  str2double(C{6}{i}(1:2));
    sec_v =  str2double(C{7}{i}(1:2));
    time_v = [time_v; hour_v*3600 + min_v*60 + sec_v];
end

% Get reward
reward_v = [];
for i = 1:epoch
    reward_v = [reward_v; str2double(C{10}{i}(1:end-1))];
end

% Get mean reward
mean_reward_v = [];
for i = 1:epoch
    mean_reward_v = [mean_reward_v; str2double(C{16}{i})];
end

time_v = seconds(time_v)
time_v.Format = 'hh:mm:ss'

%% plot
figure(1)
hold on
grid on
plot(time_v,reward_v)