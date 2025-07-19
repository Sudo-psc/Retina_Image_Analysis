# Contributing to Retina Image Analysis / Contribuindo para o Projeto

## Getting Started / Começando

### Prerequisites / Pré-requisitos

- Python 3.9+
- Poetry for dependency management
- Git for version control

### Development Setup / Configuração de Desenvolvimento

1. **Clone the repository / Clone o repositório**
   ```bash
   git clone https://github.com/username/Retina_Image_Analysis.git
   cd Retina_Image_Analysis
   ```

2. **Install Poetry / Instale o Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies / Instale as dependências**
   ```bash
   poetry install
   ```

4. **Activate virtual environment / Ative o ambiente virtual**
   ```bash
   poetry shell
   ```

5. **Install pre-commit hooks / Instale os hooks de pre-commit**
   ```bash
   pre-commit install
   ```

## Development Workflow / Fluxo de Desenvolvimento

### Branch Strategy / Estratégia de Branches

- `main`: Production-ready code / Código pronto para produção
- `develop`: Integration branch / Branch de integração
- `feature/*`: Feature development / Desenvolvimento de funcionalidades
- `bugfix/*`: Bug fixes / Correção de bugs
- `hotfix/*`: Urgent production fixes / Correções urgentes em produção

### Making Changes / Fazendo Alterações

1. **Create a feature branch / Crie uma branch de funcionalidade**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes / Faça suas alterações**
   - Write code following the project standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests / Execute os testes**
   ```bash
   pytest
   ```

4. **Check code quality / Verifique a qualidade do código**
   ```bash
   black .
   flake8
   mypy src/
   ```

5. **Commit your changes / Commit suas alterações**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

6. **Push and create PR / Push e crie um PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Standards / Padrões de Código

### Python Style Guide / Guia de Estilo Python

- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints for all functions
- Write descriptive docstrings

### Commit Message Convention / Convenção de Mensagens de Commit

```
type(scope): description

body (optional)

footer (optional)
```

**Types / Tipos:**
- `feat`: New feature / Nova funcionalidade
- `fix`: Bug fix / Correção de bug
- `docs`: Documentation / Documentação
- `style`: Formatting / Formatação
- `refactor`: Code refactoring / Refatoração
- `test`: Adding tests / Adicionando testes
- `chore`: Maintenance / Manutenção

### Testing Guidelines / Diretrizes de Teste

- Write unit tests for all new functions
- Aim for >90% test coverage
- Use descriptive test names
- Mock external dependencies
- Test edge cases and error conditions

## Project Structure / Estrutura do Projeto

```
src/
├── data/          # Data processing modules
├── models/        # ML models and architectures
├── training/      # Training scripts
├── inference/     # Inference modules
├── api/           # REST API
└── utils/         # Utility functions
```

## Code Review Process / Processo de Revisão de Código

1. Create pull request with clear description
2. Ensure all tests pass
3. Request review from at least one team member
4. Address feedback and suggestions
5. Merge after approval

## Documentation / Documentação

- Update README.md for major changes
- Add docstrings to all public functions
- Update API documentation
- Include examples in documentation

## Getting Help / Obtendo Ajuda

- Check existing issues and discussions
- Create an issue for bugs or feature requests
- Join our community discussions
- Contact maintainers for urgent matters

## License / Licença

By contributing, you agree that your contributions will be licensed under the MIT License.
