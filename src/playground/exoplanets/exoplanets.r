p = read.csv("C://xampp//htdocs//gpu_freedom//src//playground//exoplanets/exoplanet.csv", header = TRUE, sep = ",") 

# fields of csv
head(p)

# average mass in Iupiter masses
mean(p$mass,  na.rm=TRUE)
sd(p$mass, na.rm=TRUE)

# average radius in Iupiter radii
mean(p$radius, na.rm=TRUE)
sd(p$radius, na.rm=TRUE)
 
# average period in days
mean(p$orbital_period, na.rm=TRUE)
sd(p$orbital_period, na.rm=TRUE)