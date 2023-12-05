%% Evaluate the learned graph
function res = evaluator(evaluator_type, varargin)
    %% metric(POL): recall, precision Fscore, NMI
    if evaluator_type == 1
        graph_real =  varargin{1};
        graph_learned = varargin{2};
        node_num = varargin{3};
        if length(size(graph_real)) == 2
            w_real = squareform(graph_real);
            w_learned = squareform(graph_learned);
        else
            w_learned = graph_learned;
            w_real = graph_real;  
        end
        
        num_variable = (node_num)*(node_num-1)/2;


        TP = 0;
        TN = 0;
        FP = 0;
        FN = 0;
        for j = 1:num_variable
            if w_real(j) ~= 0 && w_learned(j) ~= 0
                TP = TP + 1;
            elseif w_real(j)== 0 && w_learned(j) == 0
                TN = TN + 1;
            elseif w_real(j) ~= 0 && w_learned(j) == 0
                FN = FN +1;
            elseif w_real(j) == 0 && w_learned(j) ~= 0
                FP = FP + 1;
            end

        end

        precision =  TP/(TP + FP + 0.0001);
        recall = TP/(TP + FN + 0.0001);
        mcc = (TP * TN - FP * FN) / (sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))+0.0001);
        f1 = 2 * precision *  recall/(precision + recall + 0.0001);
  
        res=[precision,recall,f1,mcc]; 
    

        
    %% metric: DOG of difference
    elseif evaluator_type ==2
        W_real =  varargin{1};
        W_learned = varargin{2};
        Z = varargin{3};
        %covert to Laplacian matrix 
        L_real  = diag(sum(W_real,2)) - W_real;
        L_learned  = diag(sum(W_learned,2)) - W_learned;
        
        
   
        diff = norm(Z'*L_real*Z - Z'*L_learned*Z,'fro');
 
        res = diff;
     
        
    %% metric:  Clustering error
     elseif evaluator_type == 3 
         labels = varargin{1};
         clustering = varargin{2};
         n=length(labels);
         
        if sum(size(labels)==[n,1])==2
            labels=reshape(labels,[1,n]);
        end

        if sum(size(clustering)==[n,1])==2
            clustering=reshape(clustering,[1,n]);
        end

        aa=unique(labels);
        J=length(aa);

        bb=unique(clustering);
        K=length(bb);

        if sum(aa==(1:J))<J
            labels_old=labels;
            temp=1;
            for ell=aa
                labels(labels_old==ell)=temp;
                temp=temp+1;
            end
        end

        if sum(bb==(1:K))<K
            clustering_old=clustering;
            temp=1;
            for ell=bb
                clustering(clustering_old==ell)=temp;
                temp=temp+1;
            end
        end



        permut=perms(1:max(K,J));
        Kfac=size(permut,1);

        error=Inf;
        clustering_temp=clustering;

        for ell=1:Kfac
            for mmm=1:K
                clustering_temp(clustering==mmm)=permut(ell,mmm);
            end    
            error_temp=sum(clustering_temp~=labels)/n;
            if error_temp<error
                error=error_temp;
            end
        end
        res = error;
         
        
    %% Balance    
    elseif evaluator_type ==4
       sensitivity = varargin{1};
       clustering = varargin{2};
       n=length(sensitivity);
         
       if sum(size(sensitivity)==[n,1])==2
            sensitivity=reshape(sensitivity,[1,n]);
       end

       if sum(size(clustering)==[n,1])==2
            clustering=reshape(clustering,[1,n]);
       end
       
        aa=unique(sensitivity);
        num_S=length(aa);

        bb=unique(clustering);
        num_C=length(bb);
        balance = 0;
        for i = 1:num_C
            indx_Ci = find(clustering==i);
            attr_in_Ci = sensitivity(indx_Ci);
            balance_i = Inf;
            for j = 1 : num_S
                for k = 1 : num_S
                    if k ~= j
                        num_attr_j = sum(attr_in_Ci == j);
                        num_attr_k = sum(attr_in_Ci == k);
                        tmp = num_attr_j/num_attr_k;
                        if tmp< balance_i
                            balance_i = tmp;  
                        end           
                    end
                    
                    
                end  
            end
            balance = balance  + balance_i;
            
            
        end
        balance_avg = balance / num_C;
        res = balance_avg;
   
        
   %% Ratio Cut
    elseif evaluator_type == 5
        W_real = varargin{1};
        node_num = size(W_real,1);
        clustering = varargin{2};
        cluster_num = varargin{3};
        all_node_idx = linspace(1,node_num,node_num);
        
        ratio_cut = 0;
        for k = 1 :cluster_num
            cluster_idx = find(clustering==k);
            is_in_cluster = ismember(all_node_idx, cluster_idx);
            out_cluster_idx = all_node_idx(~is_in_cluster); 
            W_ooc = W_real(cluster_idx, out_cluster_idx);
            ratio_cut_k = sum(sum(W_ooc))/length(cluster_idx);
            ratio_cut = ratio_cut + ratio_cut_k;   
        end
        res = ratio_cut;
        
        

    end
    
   
   

end


function NMI = calNMI(A,B)
    H=accumarray([A B],ones(1,size(A,1)));
    Pab=H/length(A);
    pa=sum(Pab,2);
    pb=sum(Pab,1);
    Pa=repmat(pa,1,size(Pab,2));
    Pb=repmat(pb,size(Pab,1),1);
    MI=sum(sum(Pab.*log2((Pab+eps)./(Pa.*Pb+eps)+eps)));
    Ha=-sum(pa.*log2(pa+eps));
    Hb=-sum(pb.*log2(pb+eps));
    NMI=2*MI/(Ha+Hb);
end







