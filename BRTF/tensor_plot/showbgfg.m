
function showbgfg(V, X, S)

figure('position', [200, 300, 600, 200]);


if ndims(X)==4
    nFrames = size(X,4);
    S = (S - min(S(:)))/(max(S(:))-min(S(:)));
    
    for i=1:nFrames
        subplot(1,3,1);  imshow(uint8(V(:,:,:,i)));
        subplot(1,3,2); imshow(uint8(X(:,:,:,i))); title(['Frame: ', num2str(i)],'Color','b','FontSize',12);
        subplot(1,3,3); imshow(S(:,:,:,i));
        pause(0.01);
    end
    
else
    nFrames = size(X,3);
    S = (S - min(S(:)))/(max(S(:))-min(S(:)));
    for i=1:nFrames
        subplot(1,3,1);  imshow(uint8(V(:,:,i)));
        subplot(1,3,2); imshow(uint8(X(:,:,i)));
        subplot(1,3,3); imshow(S(:,:,i),[]);
        pause(0.01);
    end
    
end

disp('Done');