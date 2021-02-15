library(shiny)

# Define UI ----
ui <- fluidPage(
  titlePanel("censusVis"),
  sidebarLayout(
    sidebarPanel(
      helpText("Create demographic maps with information from the 2010 US Census"),
      selectInput(
        "selectvar",
        label = "Choose a Variable to Display",
        choices = list(
          "Percent White",
          "Percent Black",
          "Percent Asian",
          "Percent Hispanic"
        )
        ,selected = "Percent Asian"
      ),
      sliderInput(
        "Slider",
        label = "Range of Interest",
        min = 0,
        max = 100,
        value = c(10, 90)
      )
    ),
    mainPanel( 
      p(textOutput("tx1"),span(textOutput("tx2"), style = "color:red;font-weight:bold;")),
      textOutput("tx3"),
      p("p creates a paragraph of text."),
      p(
        "A new p() command starts a new paragraph. Supply a style attribute to change the format of the entire paragraph.",
        style = "font-family: 'times'; font-si16pt"
      ),
      strong("strong() makes bold text."),
      em("em() creates italicized (i.e, emphasized) text."),
      br(),
      code("if patates
            kizart
           else if pirinc
           pilav"),
      div(
        "div creates segments of text with a similar style. This division of text is all blue because I passed the argument 'style = color:blue' to div",
        style = "color:blue"
      ),
      br(),
      p(
        "span does the same thing as div, but it works with",
        span("groups of words", style = "color:blue"),
        "that appear inside a paragraph."
      )
    )
  )
)

# Define server logic ----
server <- function(input, output) {
  output$tx1 <- renderText('You have selected: ')
  output$tx2 <- renderText(input$selectvar)
  output$tx3 <- renderText(paste('You have chosen a range that goes:',input$Slider[1], " to", input$Slider[2]))
}

# Run the app ----
shinyApp(ui = ui, server = server)