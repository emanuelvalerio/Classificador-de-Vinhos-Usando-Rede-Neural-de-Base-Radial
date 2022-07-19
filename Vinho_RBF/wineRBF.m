
base = readmatrix( 'wine.data' , 'FileType' , 'text' );

newClasses = zeros(N,3);

for i = 1:N
    if base(i,1) == 1
        newClasses(i,1) = 1;
    elseif base(i,1) == 2
        newClasses(i,2) = 1;
    elseif base(i,1) == 3
        newClasses(i,3) = 1;
    end
end

base = [base newClasses];
N = length(base);
q = 25; % Numero de Neuronios
p = 13 ; % numero de atributos
cont = 0;

for k = 1:N
    x = base(:,2:14)';   % x é a matriz de atributos
    % Por conta da sensibilidade da rede neural RBF, temos que normalizar a
    % base de dados
    for i = 1:p
        x(i,:) = (x(i,:)-mean(x(i,:)))/std(x(i,:)); % Normalizando os atributos
    end

    % Aplicando o método Leave-on-out

    x_teste = x(:,k);   % Adiciona a base de teste
    x(:,k) = [];        % Exclui o dado da base de treino
    D = base(:,15:17)'; % Matriz das classes binarizadas
    d_teste = D(:,k);
    D(:,k) = [];
 
    c = randn(p,q); % Geração dos centroides funcao randn()  gera números aleatórios normalmente distribuídos;

% ******* Calculo de uma estimativa para o desvio padrao (sigma) ******** %

% O método para estimar o desvio padrao é baseado na média da distância
% mínima entre os centros e as médias das somas das distâncias de todos os
% centros.

    sum_total = 0;
    mini = norm(c(2,:) - c(1,:)); % Minimo valor inicial
    for i = 1:p  
        sum_center = 0;
        for j = i+1:p
                sum_center = sum_center +(sum(norm(c(j,:) - c(i,:)))); % Soma da diferença dos centros
             if j == p
                 sum_center = sum_center/j;
             end
            if norm(c(j,:) - c(i,:)) < mini
                mini = norm(c(j,:) - c(i,:));
            end
        end
      sum_total = sum_total + sum_center;
    end

  sigma = (1/2)*(sum_total+mini);
  %sigma =  22.5639;
% *****************************************************************************%
    z = zeros(q,N-1);

     for i = 1:N-1
         for j = 1:q
             u = norm(x(:,i)-c(:,j));
              z(j,i) =  exp(-u^2/(2*sigma^2));  % Funcao de base radial (gaussiana); 
         end
     end

      z = [(-1)*ones(1,N-1);z]; % Adicionando o bias
  
       M = D*z'*(z*z')^(-1);

   % Realizando o teste, com a base de dados que não foi usada no treino;
     
   z_teste = zeros(1,q);
    for i = 1:q
        u = norm(x_teste-c(:,i));
        z =  exp(-u^2/(2*sigma^2));
        z_teste(i) = z;
    end
      z_teste = [(-1) ; z_teste']; % Acrescentando o Bias a z_teste
    
       saida  = M*z_teste;         % Resultado da camada de saída
     
     % Comparando os máximos valores, d_teste que é o valor esperado e
     % saida que o resultado obtido pela RBFNN

      [a b] = max(d_teste);
      [c d] = max(saida);
      
      if b==d % Calculo da quantidade de acertos da rede neural;
          cont = cont+1;
      end
 end 

fprintf('Accuracy : %.5f\n',cont/N); 