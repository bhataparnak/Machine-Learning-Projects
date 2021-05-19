[Tlabels, Tfeatures] = libsvmread('data.txt');
svm_classifier = svmtrain(Tlabels, Tfeatures, '-s 0 -t 0 -c 5');
pos_count = find(Tlabels == 1);
neg_count = find(Tlabels == -1);
svm = load('svm_data.txt');
w = svm_classifier.SVs' * svm_classifier.sv_coef;
b = -svm_classifier.rho;
if (svm_classifier.Label(1) == -1)
    w = -w; b = -b;
end
plot(svm(pos_count, 2), svm(pos_count,3), '+r')
hold on
plot(svm(neg_count, 2), svm(neg_count,3), 'xg')
xp = linspace(min(Tfeatures(:,1)), max(Tfeatures(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
plot(xp, yp, '-b');