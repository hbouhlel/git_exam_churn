# git_exam_churn



1.	Création Reposotery GIT : 
Nom attribué est git_exam_churn
https://github.com/hbouhlel/git_exam_churn

2.	Création clé SSH GIT pour Authentification :
a.	Générer une clé SSH
   		 ssh-keygen -t rsa -b 4096 -C hbouhlel@gmail.com
Le clé enregistré sous /home/ubuntu/.ssh/id_rsa

b.	Récupération du code Public
Sous le repertoire .ssh on récupére le code avec la commande
ls –a
cd .ssh
ls 
cat id_rsa.pub
LE Code Public est copié dans le GIT dans la partie SH and GPG keys
3.	Clonage du reposotory git :
git clone git@github.com:/hbouhlel/git_exam_churn.git

4.	Ajout des fichier dans le Github
File Python+file csv et mise à jour du dossier
Git pull
git


5.	Création du fichier Dockerfile
a.	On install les différents library qu’on a besoin
b.	On définit le WORKDIR
c.	On définit le port
d.	On définit le fichier python à exécuter

6.	Création du fichier Docker-compose.yml 
a.	Créer un premier Docker à partir de l’image datascientest/fastapi:1.0.0
b.	Créer un deuxième docker my_exam_scoring_container  à partir de l’image image_exam_scoring  avec dépendance (depends_on) du premier docker 
c.	Création de my_network
7.	Création Setup.sh :
o Création mon image image_exam_scoring
o Lancer docker Compose
8.	Mise à jour du GitHub après création d’une nouvelle branch
Git fetch
git checkout new_branch
git push -u origin new_branch


9.	Exécuter la commande setup
bash setup.sh

