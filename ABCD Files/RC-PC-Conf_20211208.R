setwd("~/")
'%!in%' = Negate('%in%')

# 10/14/2021 package set-up modification
# 11/19/2021 I can't seem to find a way to use packages without installing them outright. 
# The below method makes the install invisible, but still requires an install

packages <- c("conflr", "readxl", "purrr", "textreadr")

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Load packages
invisible(lapply(packages, library, character.only = TRUE))

RCconnection <- redcapAPI::redcapConnection(url=Sys.getenv("REDCAP_URL"), token=Sys.getenv("REDCAP_TOKEN"))

# You can select instruments to export two ways
# I need to include a prompt that reminds a user that the input Excel file should have a column named
# "form" that holds the REDcap names of the instrument, and a column named "unique_event_name" if they are 
# trying to use the entire script.
# I will make some of the filters optional, but I am considering removing the input option for an Excel sheet
# just because we don't know what will be on it, or if it will be correctly aligned with the REDCap mappings 

print("If you would like to input your own Excel file, please write: spreadsheet")
data_input_method = readline("otherwise, enter any other string")


## The below if() statement checks an input excel spreadsheet for the characteristics
## required to pull instruments from REDCap. These characteristics exist automatically
## in dataframes provided by the exportMappings() function, but if there is a specific
## list of instruments to be moved, I have included this option.

if(data_input_method == "spreadsheet"){
  valid_sheet = FALSE
  
  input_excel_filepath = readline("Enter the path to your spreadsheet of instruments to be extracted") 
  
  
  while(valid_sheet == FALSE){
    
    
    if(file.exists(input_excel_filepath)){
      all_insts_df = readxl::read_excel(input_excel_filepath)
    }
    else{
      print("Filepath invalid, please ensure it is not bounded by quotations")
      ss_sure_1 = askYesNo("Are you sure you would like to input a spreadsheet?")
      
      if(ss_sure_1 == TRUE){
        input_excel_filepath = readline("Enter the path to your spreadsheet of instruments to be extracted") 
        all_insts_df = readxl::read_excel(input_excel_filepath)
        
      }else{
        all_insts_df = redcapAPI:: exportMappings(RCconnection) 
        valid_sheet = TRUE
      }
    }
    
    if("form" %in% colnames(all_insts_df) && "unique_event_name" %in% colnames(all_insts_df)){
      valid_sheet = TRUE
      
    }else{
      print("Spreadsheet format is invalid. Please ensure that there is a column labled ( form ) 
          and a column labled ( unique_event_name )")
      
      ss_sure_2 = askYesNo("Are you sure you would like to input a spreadsheet?")
      
      if(ss_sure_2 == TRUE){
        input_excel_filepath = readline("Enter the path to your spreadsheet of instruments to be extracted") 
        all_insts_df = readxl::read_excel(input_excel_filepath)
        
      }else{
        all_insts_df = redcapAPI:: exportMappings(RCconnection) 
        valid_sheet = TRUE
      }
    }
  }
  
  ## This else follows the initial if() statemtent
}else{
  all_insts_df = redcapAPI:: exportMappings(RCconnection)  
}



# record a filepath to store the PDFs temporarily
print("if you do not enter a filepath, the PDFs will be temporarily stored in your working directory")
Filepath = readline(prompt = 
                      "Enter the path to the folder where 
REDCap PDFs should be stored.
Make sure the path is not bounded by quotations,
as this may result in an error:   ") 



## 11/19/2021  need to write error catcher for filepaths that do not exist
while(file.exists(Filepath) == FALSE){
  Filepath = readline(prompt = "path not valid, ensure path is not enclosed by quotations")
}

if (file.exists(Filepath)){
  setwd(Filepath)
}






## we want to see what options for filtering we have by examining the output
## This loop may take a bit of time, but it displays the names of unique events
## without duplication, which is not acheivable from the raw data.
event_names = c()

for(i in all_insts_df$unique_event_name){
  if(i %!in% event_names){
    event_names = c(event_names, i)
  }
  
}

print("Below is a list of events appearing in the database")
print(event_names)





# Select the event from which you would like to retrieve PDFs
# This selection could be expanded to multiple events

input_event = event_names[strtoi(readline(prompt = "type the index you wish to select"))] 
correct = FALSE

while(correct == FALSE){
  
  if(input_event %in% event_names){
    print(paste("you have chosen", input_event))
    correct = askYesNo("Is this correct?") 
    
    if (correct == FALSE){
      print(event_names)
      input_event = event_names[strtoi(readline(prompt = "type the index you wish to select"))] 
    }
    
  }else{
    print(event_names)
    print("index not recognized, please select the numerical index of the event you wish to transfer")
    input_event = event_names[strtoi(readline(prompt = "type the index you wish to select"))] 
  }
}





## This function will be mapped onto a single column of the data frame created by exportMappings()
## It will return the form name of all instruments at input_event 


event_filter_insts <- function(event_name, inst_info){
  
  ## if the event appears in the instrument list:
  if(event_name %in% inst_info$unique_event_name){
    
    ## create a vector of all instruments related to the input event
    filtered_insts = c(inst_info$form[inst_info$unique_event_name == event_name])
    
    return (filtered_insts)
  }
  else{
    print("error, [event-name] not found in the input list")
  }
}

## we run the above function for our selected event [input_event] and the input 
## spreadsheet or [exportMappings] call
target_insts = event_filter_insts(input_event, all_insts_df)






## The below pulls the PDFs from REDCap
## we want to create a new directory in which to store these files.
## We will delete this newly-created directory after uploading the PDFs to confluence

## create a directory
temp_dir = paste("RC_PDFs_", Sys.Date(), sep = '')
dir.create(temp_dir)
setwd(paste(getwd(), temp_dir, sep = '/'))


## a list of filepaths that will be used to post the REDCap PDFs to Confluence.

# first we define a function that retrieves the PDFs
get_PDF <- function(PDF_name){
  redcapAPI::exportPdf(RCconnection,
                       getwd(),
                       filename = PDF_name,
                       record = NULL,
                       events = input_event,
                       instruments = PDF_name,
                       all_records = FALSE)
}


# Then we run the function for all values of target_insts.
# This will store the PDFs to a freshly created directory, which we can delete 
# later. 

purrr:: map(target_insts[], get_PDF)


# This line will return the contents of the current directory, which we set to the temporary folder above.
instrument_pdfs_list <- list.files()



# The ID of a Confluence page can be found by selecting the three dots in the upper right
# and selecting "Page Information." The ID appears in the URL of each page created by clicking the three dots.


page_id = readline("Please enter the ID of the Confluence page you wish to upload these files to. 
This ID is the digit sequence in the URL found by selcting the [Page Information] 
                   option from the upper right corner.")




lst_attach = conflr::confl_list_attachments(page_id, limit = 10^4)


## This function updates existing PDFs on Confluence. This function will be applied to all entries of the 
## 'Results' return from the list_attachments() function.
update_info <- function(result_num){
  if(result_num$title %in% instrument_pdfs_list){
    conflr:: confl_update_attachment_metadata(id = page_id, filename = result_num$title, attachmentId = result_num$id)
  }
  return(result_num$title)
}

## This line creates a list of instruments already in Confluence (so they do not get uploaded twice)
## and updates the PDF data of these instruments. 
existing_insts = purrr:: map(lst_attach$results, update_info)



## The below uses the 'existing_insts' variable we created in the previous map() call to exclude
## existing files from being uploaded twice
post_to_confl <- function(att){
  if(att %!in% existing_insts){
    conflr:: confl_post_attachment(page_id, att)
  }
}

purrr :: map(instrument_pdfs_list[], post_to_confl)




## Asks user if the PDFs from REDCap should be removed from personal storage
delete = askYesNo("Would you like to remove the PDF files from your PC?") 

if(delete == TRUE){
  setwd(Filepath)
  file.remove(temp_dir)
  unlink(temp_dir, recursive = TRUE)
}
if(file.exists(temp_dir) == FALSE){
  print("ignore warning message, the file has been removed")
}

