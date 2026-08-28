use std::io;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs},
    Terminal,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Merge,
    Quantize,
    Eval,
    Models,
    Settings,
}

impl Tab {
    fn all() -> Vec<Tab> {
        vec![Tab::Merge, Tab::Quantize, Tab::Eval, Tab::Models, Tab::Settings]
    }

    fn title(&self) -> &str {
        match self {
            Tab::Merge => "Merge",
            Tab::Quantize => "Quantize",
            Tab::Eval => "Eval",
            Tab::Models => "Models",
            Tab::Settings => "Settings",
        }
    }
}

struct App {
    current_tab: usize,
    should_quit: bool,
}

impl App {
    fn new() -> Self {
        Self {
            current_tab: 0,
            should_quit: false,
        }
    }

    fn next_tab(&mut self) {
        self.current_tab = (self.current_tab + 1) % Tab::all().len();
    }

    fn prev_tab(&mut self) {
        if self.current_tab == 0 {
            self.current_tab = Tab::all().len() - 1;
        } else {
            self.current_tab -= 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();

    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(1),
                ])
                .split(f.area());

            // Tabs
            let all_tabs = Tab::all();
            let titles: Vec<Line> = all_tabs.iter().map(|t| {
                Line::from(Span::styled(
                    t.title(),
                    if all_tabs[app.current_tab] == *t {
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::White)
                    },
                ))
            }).collect();

            let tabs = Tabs::new(titles)
                .block(Block::default().borders(Borders::ALL).title("Forge"));
            f.render_widget(tabs, chunks[0]);

            // Content
            let content = match Tab::all()[app.current_tab] {
                Tab::Merge => Paragraph::new(vec![
                    Line::from("Model Merging"),
                    Line::from(""),
                    Line::from("Supported methods: linear, slerp, ties, dare, della,"),
                    Line::from("passthrough, darwin, frankenmerge"),
                    Line::from(""),
                    Line::from("Use CLI: forge merge --help"),
                ]),
                Tab::Quantize => Paragraph::new(vec![
                    Line::from("Quantization"),
                    Line::from(""),
                    Line::from("Methods: jang, dynamic3, apex, btl4, mixed"),
                    Line::from(""),
                    Line::from("Use CLI: forge quantize --help"),
                ]),
                Tab::Eval => Paragraph::new(vec![
                    Line::from("Evaluation"),
                    Line::from(""),
                    Line::from("Benchmarks: hella, mmlu, arc, gsm8k, gpqa"),
                    Line::from("Evals: ace, swe, terminal, gaia, hle"),
                    Line::from(""),
                    Line::from("Use CLI: forge eval --help"),
                ]),
                Tab::Models => Paragraph::new(vec![
                    Line::from("Model Browser"),
                    Line::from(""),
                    Line::from("Use CLI: forge search \"model name\""),
                ]),
                Tab::Settings => Paragraph::new(vec![
                    Line::from("Settings"),
                    Line::from(""),
                    Line::from("Hardware detection: forge info --hardware"),
                ]),
            };

            let content = content.block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(match Tab::all()[app.current_tab] {
                        Tab::Merge => "Merge",
                        Tab::Quantize => "Quantize",
                        Tab::Eval => "Eval",
                        Tab::Models => "Models",
                        Tab::Settings => "Settings",
                    }),
            );
            f.render_widget(content, chunks[1]);

            // Status bar
            let status = Paragraph::new(Line::from(vec![
                Span::styled(" Tab/Shift+Tab: navigate | q: quit ", Style::default().fg(Color::DarkGray)),
            ]));
            f.render_widget(status, chunks[2]);
        })?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => app.should_quit = true,
                        KeyCode::Tab => app.next_tab(),
                        KeyCode::BackTab => app.prev_tab(),
                        _ => {}
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
