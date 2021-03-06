library("shiny")

#runExample("01_hello")


# Define UI for app that draws a histogram ----
ui <- fluidPage(
  
  # App title ----
  #titlePanel("Hello Shiny!"),
  titlePanel("Merhaba la Parlak!"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Slider for the number of bins ----
      sliderInput(inputId = "bins",
                  label = "Number of bins:",
                  min = 1,
                  max = 50,
                  value = 30)
      
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Histogram ----
      plotOutput(outputId = "distPlot")
      
    )
  )
)
# Define server logic required to draw a histogram ----
server <- function(input, output,session) {
  
  # Histogram of the Old Faithful Geyser Data ----
  # with requested number of bins
  # This expression that generates a histogram is wrapped in a call
  # to renderPlot to indicate that:
  #
  # 1. It is "reactive" and therefore should be automatically
  #    re-executed when inputs (input$bins) change
  # 2. Its output type is a plot
  # 
  
  output$distPlot <- renderPlot({
    
    x    <- faithful$waiting
    bins <- seq(min(x), max(x), length.out = input$bins + 1)
    


    invalidateLater(500)
    secs <- as.numeric(format(Sys.time(), "%S"))
    
    if(secs %% 2 == 1){
      hist(x, breaks = bins, col = "#FFAA00", border = "white",
         xlab = "Waiting time to next eruption (in mins)",
         main = "Histogram of waiting times")}
    else
    {
      hist(x, breaks = bins, col = "#75AADE", border = "white",
           xlab = "Waiting time to next eruption (in mins)",
           main = "Histogram of waiting times")
      }

  

    
  })
  
}



# See above for the definitions of ui and server

shinyApp(ui = ui, server = server)