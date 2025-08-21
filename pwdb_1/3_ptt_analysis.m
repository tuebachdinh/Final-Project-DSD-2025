function part3_ptt_analysis(data, plaus_idx, age)
%PART3_PTT_ANALYSIS Insights about time delay and PTT analysis

onsets = data.waves.onset_times;
onset_aortic = onsets.P_AorticRoot(plaus_idx);
onset_radial = onsets.P_Radial(plaus_idx);
PTT_aor_to_rad = onset_radial - onset_aortic;

fprintf('Mean PTT (aortic root to radial): %.3f s (std: %.3f s)\n', ...
    mean(PTT_aor_to_rad), std(PTT_aor_to_rad));

pw_inds = data.pw_inds;
PTT_pwinds = pw_inds.Radial_PTT(plaus_idx);
PTT_pwinds_positive = PTT_pwinds(PTT_pwinds >= 0);
fprintf('PTT (from pw_inds, positive only): mean %.3f s (N = %d)\n', ...
    mean(PTT_pwinds_positive), numel(PTT_pwinds_positive));

figure;
histogram(PTT_pwinds_positive, 40);
xlabel('PTT (AorticRoot \rightarrow Radial) [s]');
ylabel('Count');
title('Distribution of Pulse Transit Time (PTT) from Heart to Wrist');
grid on;

unique_ages = unique(age);
figure;
boxplot(PTT_aor_to_rad, age, 'Labels', string(unique_ages));
xlabel('Age (years)');
ylabel('PTT (s)');
title('PTT (Heart to Wrist) vs Age');
grid on;

fprintf('Part 3: PTT analysis completed\n');
end