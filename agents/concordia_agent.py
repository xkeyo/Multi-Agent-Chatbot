from ollama_service import ask_ollama

def concordia_agent(message: str) -> str:
    # Comprehensive additional context extracted and synthesized from:
    # https://www.concordia.ca/academics/undergraduate/computer-science.html
    additional_context = """
Additional Context for Concordia University Computer Science Admissions:

Program Overview:
- The Bachelor of Computer Science (BCompSc) is a comprehensive degree offered by Concordia University through the Gina Cody School of Engineering and Computer Science.
- The program is designed to provide a solid foundation in both theoretical and practical aspects of computer science, including programming, algorithms, data structures, computer architecture, operating systems, and software engineering.
- Depending on a student’s academic background, the degree requires between 90 to 120 credits of full-time study over three to four years.

Curriculum & Structure:
- Core courses cover fundamental topics such as mathematical basics, programming methodology, and system hardware, ensuring that graduates have a broad and deep understanding of computer science.
- In addition to the core, students complete complementary core courses that include technical writing, communication, and an exploration of the social and ethical dimensions of information and communication technologies.
- Electives allow students to specialize in areas of interest (e.g., artificial intelligence, data analytics, web services) and can be chosen from Computer Science electives as well as Mathematics electives.
- Many students have the option to complete joint majors (for example, in Data Science or Computation Arts) or minors to tailor their education further.

Admission Criteria:
- For Quebec CEGEP applicants, admission is based on an overall average of 27 and a minimum math average of 26, with prerequisite courses in Calculus and Linear Algebra.
- High school applicants are generally expected to achieve an A- overall with strong performance in mathematics.
- International and other qualification streams (such as IB, A-levels, or university transfers) have corresponding requirements that ensure students have a solid academic foundation.
- Applicants must meet Concordia’s minimum admission requirements; meeting these does not guarantee admission, as selections are also based on the applicant pool.

Co-op & Experiential Learning:
- The program is offered in a co-op format, giving students the opportunity to participate in paid work terms (typically 12 to 16 weeks) that provide practical industry experience.
- Co-op placements are arranged with a range of employers, enabling students to apply theoretical knowledge to real-world problems and enhancing their career readiness.

Career Opportunities:
- Graduates of the program pursue careers in diverse sectors such as healthcare, communications, finance, manufacturing, and technology.
- The program’s strong emphasis on both theory and practical application equips students with problem-solving skills and innovative thinking essential for success in a rapidly evolving digital landscape.
- Concordia’s strong industry connections and active student societies further enrich the learning experience and facilitate networking opportunities.

Additional Resources:
- For the most detailed and up-to-date information, applicants are encouraged to visit the official Concordia Computer Science website and consult the Undergraduate Calendar.
- The program also offers various student support services, advising, and academic resources to help students succeed throughout their academic journey.
"""
    prompt = f"""You are an expert in Concordia University Computer Science Admissions with a deep understanding of the program structure, admission criteria, co-op opportunities, and career outcomes. Use the following additional context to provide a detailed and informed answer, but do not repeat the context verbatim in your final response.

Background Information (for internal use):
{additional_context}

User: {message}
Assistant:"""
    return ask_ollama(prompt)
