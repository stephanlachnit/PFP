% Versuchsteil 1

clearvars();
[optimizer, metric] = imregconfig('monomodal');

% import pictures as grayscale
im_00cm = rgb2gray(imread('00cm.jpg'));
im_05cm = rgb2gray(imread('05cm.jpg'));
im_15cm = rgb2gray(imread('15cm.jpg'));
im_35cm = rgb2gray(imread('35cm.jpg'));
im_65cm = rgb2gray(imread('65cm.jpg'));

% set references
reference_00cm = imref2d(size(im_00cm));
reference_05cm = imref2d(size(im_05cm));
reference_15cm = imref2d(size(im_15cm));
reference_35cm = imref2d(size(im_35cm));

% find transformation parameters
T_05cm = imregtform(im_05cm, im_00cm, 'translation', optimizer, metric);
T_10cm = imregtform(im_15cm, im_05cm, 'translation', optimizer, metric);
T_15cm = imregtform(im_15cm, im_00cm, 'translation', optimizer, metric);
T_20cm = imregtform(im_35cm, im_15cm, 'translation', optimizer, metric);
T_30cm = imregtform(im_65cm, im_35cm, 'translation', optimizer, metric);
T_35cm = imregtform(im_35cm, im_00cm, 'translation', optimizer, metric);
T_65cm = imregtform(im_65cm, im_35cm, 'translation', optimizer, metric);

% translate pictures
trans_im_05cm = imwarp(im_05cm, T_05cm, 'OutputView', reference_00cm);
trans_im_10cm = imwarp(im_15cm, T_10cm, 'OutputView', reference_05cm);
trans_im_15cm = imwarp(im_15cm, T_15cm, 'OutputView', reference_00cm);
trans_im_20cm = imwarp(im_35cm, T_20cm, 'OutputView', reference_15cm);
trans_im_35cm = imwarp(im_35cm, T_35cm, 'OutputView', reference_00cm);
trans_im_30cm = imwarp(im_65cm, T_30cm, 'OutputView', reference_35cm);
trans_im_65cm = imwarp(im_65cm, T_65cm, 'OutputView', reference_00cm);

% Bilder überlagern
% 05cm: imshowpair(im_00cm, trans_im_05cm);
% 10cm: imshowpair(im_05cm, trans_im_15cm);
% 15cm: imshowpair(im_00cm, trans_im_15cm);
% 20cm: imshowpair(im_15cm, trans_im_35cm);
% 30cm: imshowpair(im_35cm, trans_im_65cm);
% 35cm: imshowpair(im_00cm, trans_im_35cm);
% 65cm: imshowpair(im_00cm, trans_im_65cm);
