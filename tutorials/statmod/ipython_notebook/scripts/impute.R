# Load Amelia library and the data
library(Amelia)
hep <- read.table('profile-data.tsv', sep = "\t", header=TRUE, na.strings="?")

# Define list of nominal (categorical) variables
nom_vars = c("OUTCOME", "VARICES", "FATIGUE", "SPIDERS", "ASCITES", "MALAISE", "HISTOLOGY")

# Generate 5 imputed data frames
imputed <- amelia(hep, m = 5, noms = nom_vars)

# Missingness map
missmap(imputed)

# Compare observed density with imputed density
compare.density(imputed, var = "PROTIME")
compare.density(imputed, var = "ALBUMIN")
compare.density(imputed, var = "ASCITES")

# "Quality" of imputation (cannot be generated for nominal variables)
overimpute(imputed, var = "PROTIME")
overimpute(imputed, var = "ALBUMIN")

# Write out the 5 imputed data frames
write.amelia(obj=imputed, file.stem="imputed-data", format="csv", row.names = FALSE)